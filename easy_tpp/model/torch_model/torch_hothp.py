import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention, EncoderLayer, attention, ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_thp import THP


class HyperbolicRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000, time_scale=1.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.time_scale = time_scale
        
        # theta_j = max_freq^(-2(j-1)/d)
        # Ranges from 1.0 down to 1/max_freq
        thetas = []
        for j in range(1, dim // 2 + 1):
            theta_j = max_freq ** (-2 * (j - 1) / dim)
            thetas.append(theta_j)

        self.register_buffer('thetas', torch.tensor(thetas))
        
        # theta_prime must be > max(theta) to ensure decay.
        # max(theta) is 1.0. So we initialize > 1.0.
        self.theta_prime = nn.Parameter(torch.tensor(1.5))

    def forward(self, time_seqs):
        # time_seqs: [batch, seq_len]
        # Scale time to prevent numerical overflow with hyperbolic functions
        scaled_time = time_seqs * self.time_scale
        
        time_seqs_expanded = scaled_time.unsqueeze(-1) # [batch, seq_len, 1]
        thetas_expanded = self.thetas.view(1, 1, -1)   # [1, 1, dim/2]
        
        # args = time * theta
        args = time_seqs_expanded * thetas_expanded
        
        cosh_args = torch.cosh(args)
        sinh_args = torch.sinh(args)
        
        # Ensure theta_prime > 1.0 (max theta)
        # Using softplus to keep it positive and adding offset
        theta_prime_val = F.softplus(self.theta_prime) + 1.0 + 1e-4
        
        return cosh_args, sinh_args, scaled_time, theta_prime_val


def apply_hyperbolic_rotary_pos_emb(q, k, cosh, sinh, time_seqs, theta_prime):
    # q, k: [batch, n_head, seq_len, d_k]
    # cosh, sinh: [batch, seq_len, dim/2] -> unsqueeze for head dim
    # time_seqs: [batch, seq_len] -> unsqueeze for head and dim
    
    cosh = cosh.unsqueeze(1) # [batch, 1, seq_len, dim/2]
    sinh = sinh.unsqueeze(1)
    
    # Expand time_seqs for broadcasting: [batch, 1, seq_len, 1]
    time_seqs_expanded = time_seqs.unsqueeze(1).unsqueeze(-1)
    
    # Split q and k into even and odd components
    q1 = q[..., 0::2]
    q2 = q[..., 1::2]
    
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]
    
    # Hyperbolic Rotation
    # Query: B(theta, m) [q1, q2]^T = [q1 cosh + q2 sinh, q1 sinh + q2 cosh]
    q1_rot = q1 * cosh + q2 * sinh
    q2_rot = q1 * sinh + q2 * cosh
    
    # Key: B'(theta, n) [k1, k2]^T = [k1 cosh - k2 sinh, -k1 sinh + k2 cosh]
    k1_rot = k1 * cosh - k2 * sinh
    k2_rot = -k1 * sinh + k2 * cosh
    
    # Exponential Decay/Growth
    # Query: exp(-m * theta')
    decay_q = torch.exp(-time_seqs_expanded * theta_prime)
    
    # Key: exp(n * theta')
    # Note: This can grow large if time is large. 
    # The dot product will be exp(-(m-n)theta'), which is safe if m > n.
    growth_k = torch.exp(time_seqs_expanded * theta_prime)
    
    # Apply factors
    q_new = torch.zeros_like(q)
    k_new = torch.zeros_like(k)
    
    q_new[..., 0::2] = q1_rot * decay_q
    q_new[..., 1::2] = q2_rot * decay_q
    
    k_new[..., 0::2] = k1_rot * growth_k
    k_new[..., 1::2] = k2_rot * growth_k
    
    return q_new, k_new


class HyperbolicRotaryMultiHeadAttention(MultiHeadAttention):
    def forward(self, query, key, value, mask, cosh=None, sinh=None, time_seqs=None, theta_prime=None, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        
        # Apply Hyperbolic Rotary Positional Embedding
        if cosh is not None and sinh is not None and time_seqs is not None and theta_prime is not None:
             query, key = apply_hyperbolic_rotary_pos_emb(query, key, cosh, sinh, time_seqs, theta_prime)

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


class HyperbolicRotaryEncoderLayer(EncoderLayer):
    def forward(self, x, mask, cosh=None, sinh=None, time_seqs=None, theta_prime=None):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, cosh=cosh, sinh=sinh, time_seqs=time_seqs, theta_prime=theta_prime))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            x = self.self_attn(x, x, x, mask, cosh=cosh, sinh=sinh, time_seqs=time_seqs, theta_prime=theta_prime)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            else:
                return x


class HoTHP(THP):
    """Torch implementation of Hyperbolic Rotary Position Embedding-based Transformer Hawkes Process (HoTHP).
       Based on HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        # Initialize TorchBaseModel directly
        TorchBaseModel.__init__(self, model_config)        
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size 
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # Hyperbolic Rotary Embedding
        # time_scale can be adjusted if timestamps are very large causing overflow
        # Here using 1.0 default, assuming normalized inputs or handled by theta_prime
        self.rotary_emb = HyperbolicRotaryEmbedding(self.d_model // self.n_head, time_scale=0.1)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.stack_layers = nn.ModuleList(
            [HyperbolicRotaryEncoderLayer(
                self.d_model,
                HyperbolicRotaryMultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
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
        # [batch_size, seq_len, dim/2]
        cosh, sinh, scaled_time, theta_prime = self.rotary_emb(time_seqs)
        
        enc_output = self.layer_type_emb(type_seqs)

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                cosh=cosh, 
                sinh=sinh,
                time_seqs=scaled_time,
                theta_prime=theta_prime)

        return enc_output
