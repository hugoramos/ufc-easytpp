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
        return F.softplus(self.theta_prime_raw) + 1 + 1e-4


def hyperbolic_attention(q, k, v, time_seqs, thetas, theta_prime, mask=None, dropout=None):
    # q, k, v tem shape [B, H, L, d_k]:
    #   B = batch size
    #   H = heads
    #   L = numero de eventos na sequência
    #   d_k = dimensão de cada head (= hidden_size / num_heads)
    d_k = q.shape[-1]

    # t=[0.0, 1.5, 4.0]:
    #            j=0   j=1   j=2
    #         ┌─────────────────────┐
    #   i=0   │  0.0  -1.5  -4.0    │ 
    #   i=1   │  1.5   0.0  -2.5    │ 
    #   i=2   │  4.0   2.5   0.0    │ 
    #         └─────────────────────┘
    #
    # unsqueeze(-1) [B, L] → [B, L, 1] 
    # unsqueeze(-2) [B, L] → [B, 1, L] 
    delta = time_seqs.unsqueeze(-1) - time_seqs.unsqueeze(-2)  # [B, L, L]

    #   cosh(x) = cosh(-x) só depende de |δ|, sinal não importa
    #   sinh(x) = -sinh(-x) depende do sinal (reintroduzido depois)
    #
    #   |δ|:              sign(δ):
    #   0.0  1.5  4.0     0  -1  -1
    #   1.5  0.0  2.5    +1   0  -1
    #   4.0  2.5  0.0    +1  +1   0
    #
    abs_delta       = delta.abs().unsqueeze(-1)    # [B, L, L, 1] a dimensão extra é para broadcast com thetas
    sign_delta      = delta.sign().unsqueeze(-1)   # [B, L, L, 1]
    thetas_expanded = thetas.view(1, 1, 1, -1)     # [1, 1, 1, d_k//2] broadcast com abs_delta

    #   cosh(x) = (eˣ + e⁻ˣ) / 2                                                                                                                          
    #   sinh(x) = (eˣ - e⁻ˣ) / 2 
    #
    #   kernel(i,j) = e^(-|δ|θ') * cosh(δ*θⱼ)  
    #                 e^(-|δ|θ') * sinh(δ*θⱼ) 

    #   e^(-|δ|θ') * cosh(|δ|*θⱼ)
    #     = e^(-|δ|θ') * (e^(|δ|θⱼ) + e^(-|δ|θⱼ)) / 2
    #     = (e^(-|δ|θ') * e^(|δ|θⱼ)  +  e^(-|δ|θ') * e^(-|δ|θⱼ)) / 2
    #     = (e^(-|δ|(θ'-θⱼ))  +  e^(-|δ|(θ'+θⱼ))) / 2 # como θ' é sempre maior que θⱼ, vai ser sempre negativo
    #        ───────────────      ───────────────
    #         exp com menos         exp com mais
    #

    #   e^(-|δ|θ') * sinh(δ*θⱼ)
    #     = e^(-|δ|θ') * sign(δ) * sinh(|δ|*θⱼ)
    #     = sinal δ * e^(-|δ|θ') * (e^(|δ|θⱼ) - e^(-|δ|θⱼ)) / 2
    #     = sinal δ * (e^(-|δ|(θ'-θⱼ)) - e^(-|δ|(θ'+θⱼ))) / 2
    #
    exp_minus  = torch.exp(-abs_delta * (theta_prime - thetas_expanded))  # e^(-|δ|(θ'-θⱼ))  [B, L, L, d_k//2]
    exp_plus   = torch.exp(-abs_delta * (theta_prime + thetas_expanded))  # e^(-|δ|(θ'+θⱼ))  [B, L, L, d_k//2]
    decay_cosh = (exp_minus + exp_plus) / 2                               # = e^(-|δ|θ') * cosh(δθⱼ)
    decay_sinh = sign_delta * (exp_minus - exp_plus) / 2                  # = e^(-|δ|θ') * sinh(δθⱼ)

    #   B(θ, δ) = e^(-|δ|θ') * [ cosh(δθ)   sinh(δθ) ]
    #                          [ sinh(δθ)   cosh(δθ) ]

    q1 = q[..., 0::2]  # dimensões pares    [B, H, L, d_k//2]
    q2 = q[..., 1::2]  # dimensões ímpares  [B, H, L, d_k//2]
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]

    # adicionado dimensões para calcular o score dos pares via broadcast
    q1_exp, q2_exp = q1.unsqueeze(3), q2.unsqueeze(3)  # [B, H, L, 1, d_k//2]
    k1_exp, k2_exp = k1.unsqueeze(2), k2.unsqueeze(2)  # [B, H, 1, L, d_k//2]

    # [q1, q2] * B * [k1, k2]ᵀ
    #   = (q1*cosh + q2*sinh)·k1 + (q1*sinh + q2*cosh)*k2
    #   = (q1*k1 + q2*k2)*cosh  +  (q1*k2 + q2*k1)*sinh
    #      ──────────────────       ──────────────────
    #           parte_cosh             parte_sinh
    #
    parte_cosh = q1_exp * k1_exp + q2_exp * k2_exp  # [B, H, L, L, d_k//2]
    parte_sinh = q1_exp * k2_exp + q2_exp * k1_exp  # [B, H, L, L, d_k//2]

    att_scores = (parte_cosh * decay_cosh.unsqueeze(1) + parte_sinh * decay_sinh.unsqueeze(1)).sum(dim=-1)  # [B, H, L, L]
    att_scores = att_scores / math.sqrt(d_k)

    if mask is not None:
        att_scores = att_scores.masked_fill(mask > 0, -1e4)

    # softmax transforma os scores em pesos que somam 1
    att_weights = torch.softmax(att_scores, dim=-1)
    if dropout is not None:
        att_weights = dropout(att_weights)

    # pondera os values pelos pesos e retorna
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


class HoTHPNoChunk(THP):
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
