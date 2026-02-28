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

        self.thetas = torch.tensor(thetas)
        # self.register_buffer('thetas', torch.tensor(thetas))

    def forward(self, t):
        # t original: [batch, seq_len]
        # t_expanded: [batch, seq_len, 1]
        t_expanded = t.unsqueeze(-1)
        
        # self.thetas: [dim/2] (vetor com as frequencias)
        # thetas_expanded: [1, 1, dim/2]
        thetas_expanded = self.thetas.view(1, 1, -1)
        
        # args = t * theta (tempo * frequência)
        # args: [batch, seq_len, dim/2]
        args = t_expanded * thetas_expanded
        
        cos_args = torch.cos(args)
        sin_args = torch.sin(args)
        
        return cos_args, sin_args


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    Aplica a rotação definida no paper (matriz de rotação R).
    R é uma matriz bloco-diagonal com blocos:
    [ cos, -sin ]
    [ sin,  cos ]
    
    Isso significa que para cada par de coordenadas (x1, x2), temos:
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos
    
    Args:
        q: [batch, n_head, seq_len, dim]
        k: [batch, n_head, seq_len, dim]
        cos: [batch, seq_len, dim]
        sin: [batch, seq_len, dim]
    """
    # Ajustamos dimensões de cos e sin para broadcasting com q e k
    # [batch, seq_len, dim/2] -> [batch, 1, seq_len, dim/2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Vamos separar os pares x1 e x2 explicitamente
    # q tem dimensão 'dim' na última coordenada.
    # Assumimos que 'dim' é par.
    # Pegamos os elementos nos índices pares (0, 2, 4...) como x1
    # e nos ímpares (1, 3, 5...) como x2
    
    # Shapes resultantes: [..., dim/2]
    q1 = q[..., 0::2]
    q2 = q[..., 1::2]
    
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]
    
    # [OTIMIZAÇÃO] Não precisamos mais fazer slicing no cos/sin.
    # Eles já vêm com o tamanho correto [dim/2] do RotaryEmbedding.forward otimizado.
    c = cos
    s = sin
    
    # Aplicamos a fórmula da rotação explicitamente:
    # x1_rot = x1 * cos - x2 * sin
    # x2_rot = x1 * sin + x2 * cos
    
    q1_rot = q1 * c - q2 * s
    q2_rot = q1 * s + q2 * c
    
    k1_rot = k1 * c - k2 * s
    k2_rot = k1 * s + k2 * c
    
    # Agora reconstruímos os tensores q e k intercalando os valores rotacionados
    # Criamos um tensor vazio com o shape original
    q_new = torch.zeros_like(q)
    k_new = torch.zeros_like(k)
    
    # Preenchemos os índices pares e ímpares
    q_new[..., 0::2] = q1_rot
    q_new[..., 1::2] = q2_rot
    
    k_new[..., 0::2] = k1_rot
    k_new[..., 1::2] = k2_rot

    # separamos o vetor inteiro em duas metades lógicas, 
    # onde cada índice $i$ em $X_1$ e $X_2$ corresponde a um par completo que precisa ser rotacionado junto. 
    #
    # Isso permite aplicar a fórmula de rotação em todos os 32 pares (se dim=64) simultaneamente 
    # com uma única operação matemática vetorizadamente.
    
    return q_new, k_new


class RotaryMultiHeadAttention(MultiHeadAttention):
    def forward(self, query, key, value, mask, cos=None, sin=None, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply RoPE if provided
        if cos is not None and sin is not None:
             query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 3) Attention
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
    # def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
    #     super(RotaryEncoderLayer, self).__init__(d_model, self_attn, feed_forward, use_residual, dropout)

    def forward(self, x, mask, cos=None, sin=None):
        if self.use_residual:
            # We pass cos, sin to self_attn via lambda
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
