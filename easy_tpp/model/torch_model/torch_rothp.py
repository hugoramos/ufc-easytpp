import torch
import torch.nn as nn
import math

from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention, EncoderLayer, SublayerConnection, ScaledSoftplus, attention
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_thp import THP


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        
        # equação 17 do RoTHP: theta_j = 10000^(-2(j-1)/d)
        thetas = []
        for j in range(1, dim // 2 + 1):
            theta_j = max_freq ** (-2 * (j - 1) / dim)
            thetas.append(theta_j)

        # self.thetas = torch.tensor(thetas) (erro no colab.. Unexpected key(s) in state_dict: "rotary_emb.thetas")
        self.register_buffer('thetas', torch.tensor(thetas))

    def forward(self, time_seqs):
        # time_seqs original: [batch, seq_len]
        # time_seqs_expanded: [batch, seq_len, 1]
        time_seqs_expanded = time_seqs.unsqueeze(-1)
        
        # self.thetas: [dim/2] (vetor com as frequencias)
        # thetas_expanded: [1, 1, dim/2]
        thetas_expanded = self.thetas.view(1, 1, -1)
        
        # args = time_seqs * theta (tempo * frequência)
        # args: [batch, seq_len, dim/2]
        args = time_seqs_expanded * thetas_expanded
        
        cos_args = torch.cos(args)
        sin_args = torch.sin(args)
        
        return cos_args, sin_args


def apply_rotary_pos_emb(q, k, cos, sin):
    # [batch, seq_len, dim/2] -> [batch, 1, seq_len, dim/2] pois q e k são [batch, n_head, seq_len, d_k]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # shapes resultantes: [..., dim/2]
    q1 = q[..., 0::2] # pra pegar os pares (x1)
    q2 = q[..., 1::2] # pra pegar os impares (x2)
    
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]
    
    # matriz de rotação R:
    # [cos, -sin]
    # [sin,  cos]
    q1_rot = q1 * cos - q2 * sin
    q2_rot = q1 * sin + q2 * cos
    
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k1 * sin + k2 * cos

    q_new = torch.zeros_like(q)
    k_new = torch.zeros_like(k)
    
    # preenchendo os índices pares e ímpares
    q_new[..., 0::2] = q1_rot
    q_new[..., 1::2] = q2_rot
    
    k_new[..., 0::2] = k1_rot
    k_new[..., 1::2] = k2_rot
    
    return q_new, k_new

class RotaryMultiHeadAttention(MultiHeadAttention):
    def forward(self, query, key, value, mask, cos=None, sin=None, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        
        # rotação dos embeddings
        if cos is not None and sin is not None:
             query, key = apply_rotary_pos_emb(query, key, cos, sin)

        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x


class RotaryEncoderLayer(EncoderLayer):
    def forward(self, x, mask, cos=None, sin=None):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, cos=cos, sin=sin))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            x = self.self_attn(x, x, x, mask, cos=cos, sin=sin)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            else:
                return x


class RoTHP(THP):
    """Torch implementation of Rotary Position Embedding-based Transformer Hawkes Process (RoTHP).
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        # Initialize TorchBaseModel directly to set up basic attributes
        TorchBaseModel.__init__(self, model_config)        
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size 
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # mudei do THP para o RoTHP
        self.rotary_emb = RotaryEmbedding(self.d_model // self.n_head)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)   # learnable mark-specific beta

        # Add MLP layer
        # Equation (5) (THP)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.stack_layers = nn.ModuleList(
            [RotaryEncoderLayer(
                self.d_model,
                RotaryMultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),
                use_residual=False, 
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, dim]
        cos, sin = self.rotary_emb(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        # nao passo temporal encoding aqui... e passo cos e sin
        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                cos=cos, 
                sin=sin)

        return enc_output
