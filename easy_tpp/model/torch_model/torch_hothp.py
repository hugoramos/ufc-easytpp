import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention, EncoderLayer, attention, ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_thp import THP


class HyperbolicRotaryEmbedding(nn.Module):
    """Adapts HoPE (Dai et al., 2025) from discrete token positions to continuous
    event timestamps in Temporal Point Processes.

    The original HoPE applies B(m*theta) to Q and B'(n*theta) to K at absolute
    positions m, n, relying on B(m)*B'(n) = B(m-n).  This identity breaks
    numerically for float32/64 when |m*theta| > ~15 because cosh and sinh grow
    exponentially and cosh^2 - sinh^2 = 1 no longer holds in finite precision.

    Instead, we compute B(delta*theta) directly from relative temporal
    distances delta = t_m - t_n inside the attention function.  This is
    algebraically equivalent to the paper's Eq. 9 but avoids catastrophic
    cancellation from absolute positions.

    theta_prime is constrained to satisfy theta' > max(theta_i) (Eq. 12),
    ensuring monotonic decay of attention scores with temporal distance.
    """

    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

        thetas = torch.tensor([
            max_freq ** (-2.0 * (j - 1) / dim) for j in range(1, dim // 2 + 1)
        ])
        self.register_buffer('thetas', thetas)

        # theta' > max(theta_i) = 1.0  (Eq. 12)
        self.theta_prime_raw = nn.Parameter(torch.tensor(0.5))

    @property
    def theta_prime(self):
        return F.softplus(self.theta_prime_raw) + self.thetas.max().item() + 1e-4


def hyperbolic_attention(query, key, value, time_seqs, thetas, theta_prime,
                         mask=None, dropout=None):
    """Scaled dot-product attention with HoPE kernel (Eq. 9 of Dai et al., 2025).

    Computes  score(m,n) = exp(-delta*theta') * (q_m^T B(delta*theta) k_n) / sqrt(d)
    where delta = t_m - t_n, directly in the relative space.

    B(delta*theta) is the Lorentz boost applied per 2D dimension pair:
        B(a) = [[cosh(a), sinh(a)],
                [sinh(a), cosh(a)]]

    This avoids the numerical instability of computing B(m)*B'(n) from
    large absolute positions, since delta is bounded by the sequence span.
    """
    B, H, L, d_k = query.shape
    half_d = d_k // 2

    # Pairwise temporal distances: delta(m,n) = t_m - t_n
    t_m = time_seqs.unsqueeze(-1)      # [B, L, 1]
    t_n = time_seqs.unsqueeze(-2)      # [B, 1, L]
    delta = (t_m - t_n)                # [B, L, L]

    # thetas: [half_d] -> [1, 1, 1, half_d]
    th = thetas.view(1, 1, 1, -1)

    # args: [B, L, L, half_d]  -- one value per (query, key, dimension-pair)
    args = delta.unsqueeze(-1) * th
    cosh_d = torch.cosh(args)          # [B, L, L, half_d]
    sinh_d = torch.sinh(args)          # [B, L, L, half_d]

    # Split Q, K into even/odd pairs
    q1 = query[:, :, :, 0::2]         # [B, H, L, half_d]
    q2 = query[:, :, :, 1::2]
    k1 = key[:, :, :, 0::2]
    k2 = key[:, :, :, 1::2]

    # For each query position m and key position n, compute:
    #   q_m^T B(delta*theta) k_n
    # = sum_j [ (q1_m*k1_n + q2_m*k2_n)*cosh(delta*theta_j)
    #         + (q1_m*k2_n + q2_m*k1_n)*sinh(delta*theta_j) ]
    #
    # q1_m: [B, H, L_q, half_d]  x  k1_n: [B, H, L_k, half_d]
    # We need the outer product over the L dimensions.
    # q1_m[..., i, :] * k1_n[..., j, :] -> [B, H, L, L, half_d]

    q1_exp = q1.unsqueeze(3)           # [B, H, L, 1, half_d]
    q2_exp = q2.unsqueeze(3)
    k1_exp = k1.unsqueeze(2)           # [B, H, 1, L, half_d]
    k2_exp = k2.unsqueeze(2)

    # Symmetric and anti-symmetric products
    sym  = q1_exp * k1_exp + q2_exp * k2_exp   # [B, H, L, L, half_d]
    asym = q1_exp * k2_exp + q2_exp * k1_exp   # [B, H, L, L, half_d]

    # cosh_d, sinh_d: [B, L, L, half_d] -> [B, 1, L, L, half_d]
    cosh_d = cosh_d.unsqueeze(1)
    sinh_d = sinh_d.unsqueeze(1)

    # Full dot product per (m, n) pair, summed over dimension pairs
    scores = (sym * cosh_d + asym * sinh_d).sum(dim=-1)  # [B, H, L, L]
    scores = scores / math.sqrt(d_k)

    # Multiplicative decay: exp(-|delta| * theta')  (Eq. 9)
    decay = torch.exp(-delta.abs().unsqueeze(1) * theta_prime)  # [B, 1, L, L]
    scores = scores * decay

    if mask is not None:
        scores = scores.masked_fill(mask > 0, -1e4)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class HyperbolicMultiHeadAttention(MultiHeadAttention):
    def forward(self, query, key, value, mask, time_seqs=None, thetas=None,
                theta_prime=None, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]

        if time_seqs is not None and thetas is not None and theta_prime is not None:
            x, attn_weight = hyperbolic_attention(
                query, key, value, time_seqs, thetas, theta_prime,
                mask=mask, dropout=self.dropout)
        else:
            x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return (self.linears[-1](x), attn_weight) if output_weight else self.linears[-1](x)
        else:
            return (x, attn_weight) if output_weight else x


class HyperbolicEncoderLayer(EncoderLayer):
    def forward(self, x, mask, time_seqs=None, thetas=None, theta_prime=None):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(
                x, x, x, mask, time_seqs=time_seqs, thetas=thetas, theta_prime=theta_prime))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            return x
        else:
            x = self.self_attn(x, x, x, mask, time_seqs=time_seqs, thetas=thetas, theta_prime=theta_prime)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            return x


class HoTHP(THP):
    """Hyperbolic Position Embedding-based Transformer Hawkes Process.

    Applies HoPE (Dai et al., 2025) to the TPP/Hawkes Process domain.
    Like RoTHP, temporal information is injected through the attention kernel
    rather than through additive temporal encoding.  HoPE uses hyperbolic
    (Lorentz) boosts instead of trigonometric rotations, producing
    monotonically decaying attention scores with temporal distance.

    Implementation note: we compute the HoPE kernel directly in relative
    time-space (delta = t_m - t_n) to avoid the catastrophic floating-point
    cancellation that occurs when applying absolute Lorentz boosts B(m), B'(n)
    with large timestamps and then relying on B(m)*B'(n) = B(m-n).
    """

    def __init__(self, model_config):
        TorchBaseModel.__init__(self, model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.hope_emb = HyperbolicRotaryEmbedding(self.d_model // self.n_head)

        self.factor_intensity_base = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.stack_layers = nn.ModuleList([
            HyperbolicEncoderLayer(
                self.d_model,
                HyperbolicMultiHeadAttention(
                    self.n_head, self.d_model, self.d_model, self.dropout,
                    output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])

    def _normalize_timestamps(self, time_seqs):
        """Map raw timestamps so that the median inter-event gap is ~1.0.

        This keeps cosh/sinh arguments bounded while preserving relative
        temporal structure, analogous to the paper's integer positions.
        """
        t_min = time_seqs.min(dim=-1, keepdim=True).values
        t_shifted = time_seqs - t_min
        diffs = t_shifted[:, 1:] - t_shifted[:, :-1]
        median_gap = diffs.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return t_shifted / median_gap

    def forward(self, time_seqs, type_seqs, attention_mask):
        norm_times = self._normalize_timestamps(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        thetas = self.hope_emb.thetas
        theta_prime = self.hope_emb.theta_prime

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                time_seqs=norm_times,
                thetas=thetas,
                theta_prime=theta_prime)

        return enc_output
