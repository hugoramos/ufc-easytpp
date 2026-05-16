import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention, EncoderLayer, attention, ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_thp import THP


class HyperbolicRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

        # frequências θⱼ = 10000^(-2(j-1)/d) — mesma fórmula do RoPE (Eqacao 17 do RoTHP)
        # cada par de dimensões (2j, 2j+1) recebe uma frequência diferente
        thetas = torch.tensor([
            max_freq ** (-2.0 * (j - 1) / dim) for j in range(1, dim // 2 + 1)
        ])
        self.register_buffer('thetas', thetas)

        # θ' é o coeficiente de decaimento global (Eqacoes 9, 10, 11 e 12 do HoPE)
        # tenho que garantir que θ' > max(θⱼ).. entan treinar outra variavel (theta_prime_raw)
        self.theta_prime_raw = nn.Parameter(torch.tensor(0.5))

    @property
    def theta_prime(self):
        # adicionadno uma mixaria pra garantir a condição da equacao 12 do HoPE
        # + max(θⱼ) + 1e-4 garante θ' > max(θⱼ)
        return F.softplus(self.theta_prime_raw) + self.thetas.max().item() + 1e-4


def hyperbolic_attention(q, k, v, time_seqs, thetas, theta_prime, mask=None, dropout=None, chunk_size=16):
    # q, k, v: [B, H, L, d_k]
    # time_seqs: [B, L]
    # thetas: [d_k//2]
    # theta_prime: scalar
    #
    # Computação em chunks ao longo da dimensão de query (i) para evitar
    # alocar tensores [B, L, L, D] completos, que com L≈250 chegam a ~1GB cada.
    # Peak por iteração ≈ O(B * H * chunk * L * D) em vez de O(B * H * L² * D).
    d_k = q.shape[-1]
    B, H, L, _ = q.shape

    # separar q e k em pares de dimensões (Eq. 5)
    q1 = q[..., 0::2]  # [B, H, L, d_k//2]
    q2 = q[..., 1::2]
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]

    # k expandido ao longo da dimensão de query — reutilizado em todos os chunks
    k1_j = k1.unsqueeze(2)  # [B, H, 1, L, d_k//2]
    k2_j = k2.unsqueeze(2)

    # tempos das key positions — fixos para todos os chunks
    t_j = time_seqs.unsqueeze(1)        # [B, 1, L]
    thetas_v = thetas.view(1, 1, -1)    # [1, 1, d_k//2]

    att_scores = torch.zeros(B, H, L, L, device=q.device, dtype=q.dtype)

    for i_start in range(0, L, chunk_size):
        i_end = min(i_start + chunk_size, L)

        # delta entre posições de query e key para este chunk
        # t_i: [B, chunk, 1]   t_j: [B, 1, L]  →  delta: [B, chunk, L]
        t_i = time_seqs[:, i_start:i_end].unsqueeze(-1)  # [B, chunk, 1]
        delta = t_i - t_j                                  # [B, chunk, L]

        abs_d  = delta.abs().unsqueeze(-1)   # [B, chunk, L, 1]
        sign_d = delta.sign().unsqueeze(-1)  # [B, chunk, L, 1]

        # kernels de decaimento hiperbólico para este chunk (Eq. 9 do HoPE)
        # ver comentário da versão anterior sobre a reformulação numérica estável
        exp_m = torch.exp(-abs_d * (theta_prime - thetas_v))  # [B, chunk, L, d_k//2]
        exp_p = torch.exp(-abs_d * (theta_prime + thetas_v))
        dc = ((exp_m + exp_p) / 2).unsqueeze(1)              # [B, 1, chunk, L, d_k//2]
        ds = (sign_d * (exp_m - exp_p) / 2).unsqueeze(1)

        # q para este chunk
        q1_i = q1[:, :, i_start:i_end, :].unsqueeze(3)  # [B, H, chunk, 1, d_k//2]
        q2_i = q2[:, :, i_start:i_end, :].unsqueeze(3)

        # (q1*k1 + q2*k2)*cosh + (q1*k2 + q2*k1)*sinh  →  soma sobre d_k//2
        # broadcast: [B, H, chunk, L, d_k//2] → sum(-1) → [B, H, chunk, L]
        att_scores[:, :, i_start:i_end, :] = (
            (q1_i * k1_j + q2_i * k2_j) * dc +
            (q1_i * k2_j + q2_i * k1_j) * ds
        ).sum(-1)

    att_scores = att_scores / math.sqrt(d_k)

    if mask is not None:
        att_scores = att_scores.masked_fill(mask > 0, -1e4)
    att_weights = torch.softmax(att_scores, dim=-1)
    if dropout is not None:
        att_weights = dropout(att_weights)
    return torch.matmul(att_weights, v), att_weights


class HyperbolicMultiHeadAttention(MultiHeadAttention):
    def forward(self, query, key, value, mask, time_seqs=None, thetas=None,theta_prime=None, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]

        #
        # diferente do rothp, nao transformo q e k
        #

        x, attn_weight = hyperbolic_attention(query, key, value, time_seqs, thetas, theta_prime,mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            return (self.linears[-1](x), attn_weight) if output_weight else self.linears[-1](x)
        else:
            return (x, attn_weight) if output_weight else x


class HyperbolicEncoderLayer(EncoderLayer):
    def forward(self, x, mask, time_seqs=None, thetas=None, theta_prime=None):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, time_seqs=time_seqs, thetas=thetas, theta_prime=theta_prime))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            x = self.self_attn(x, x, x, mask, time_seqs=time_seqs, thetas=thetas, theta_prime=theta_prime)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            else:
                return x


class HoTHP(THP):
    """Torch implementation of Hyperbolic Rotary Position Embedding-based Transformer Hawkes Process (HoTHP).
    """

    def __init__(self, model_config):
        TorchBaseModel.__init__(self, model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # mudei do RoTHP para o HoTHP
        self.hope_emb = HyperbolicRotaryEmbedding(self.d_model // self.n_head)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)

        # Add MLP layer
        # Equation (5) (THP)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.stack_layers = nn.ModuleList([
            HyperbolicEncoderLayer(
                self.d_model,
                HyperbolicMultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                    output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])

    def _normalize_timestamps(self, time_seqs):
        return time_seqs - time_seqs[:, :1]

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
