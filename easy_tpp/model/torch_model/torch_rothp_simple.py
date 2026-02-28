import torch
import torch.nn as nn
import math

# Importa componentes base para reduzir código (Herança)
from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention, EncoderLayer, ScaledSoftplus, attention
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class RotaryEmbedding(nn.Module):
    """
    Calcula os Embeddings de Posição Rotacional (RoPE).
    Gera as matrizes de rotação (cos, sin) baseadas no tempo.
    """
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        
        # Pré-calcula frequências theta
        # theta_j = 10000^(-2(j-1)/d)
        thetas = []
        for j in range(1, dim // 2 + 1):
            theta_j = max_freq ** (-2 * (j - 1) / dim)
            thetas.append(theta_j)
            
        self.register_buffer('thetas', torch.tensor(thetas))

    def forward(self, t):
        # t: [batch, seq_len] -> [batch, seq_len, 1]
        t_expanded = t.unsqueeze(-1)
        # thetas: [dim/2] -> [1, 1, dim/2]
        thetas_expanded = self.thetas.view(1, 1, -1)
        
        # Ângulo = tempo * frequência
        args = t_expanded * thetas_expanded
        
        # Calcula cos/sin e duplica para pares (x, y)
        # [cos1, cos2] -> [cos1, cos1, cos2, cos2]
        cos = torch.repeat_interleave(torch.cos(args), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(args), 2, dim=-1)
        
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Aplica a rotação aos vetores Query e Key.
    Rotação 2D: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
    """
    # Ajusta dimensões para broadcasting
    cos = cos.unsqueeze(1) # [batch, 1, seq_len, dim]
    sin = sin.unsqueeze(1)
    
    # Separa coordenadas pares (x) e ímpares (y)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    
    # Pega cos/sin correspondentes aos pares
    c, s = cos[..., 0::2], sin[..., 0::2]
    
    # Aplica rotação
    q_rot = torch.zeros_like(q)
    q_rot[..., 0::2] = q1 * c - q2 * s
    q_rot[..., 1::2] = q1 * s + q2 * c
    
    k_rot = torch.zeros_like(k)
    k_rot[..., 0::2] = k1 * c - k2 * s
    k_rot[..., 1::2] = k1 * s + k2 * c
    
    return q_rot, k_rot


class RotaryAttention(MultiHeadAttention):
    """
    Estende a MultiHeadAttention padrão para suportar RoPE.
    Reaproveita a inicialização de pesos (W_q, W_k, W_v) da classe pai.
    """
    def forward(self, query, key, value, mask, cos=None, sin=None, output_weight=False):
        # Prepara a máscara
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1. Projeções Lineares (usa self.linears da classe pai)
        # Transforma e divide em cabeças: [batch, n_head, seq_len, d_k]
        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        
        # 2. APLICA A ROTAÇÃO (O diferencial do RoTHP)
        if cos is not None and sin is not None:
             query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 3. Calcula Atenção (usa função utilitária do framework)
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 4. Reagrupa e projeta a saída
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)

        # Usa a última linear da lista (projeção de saída)
        if self.output_linear:
            return self.linears[-1](x)
        return x


class RotaryEncoderLayer(EncoderLayer):
    """
    Estende EncoderLayer padrão para passar cos/sin adiante.
    """
    def forward(self, x, mask, cos=None, sin=None):
        # Define uma função lambda para injetar cos/sin na chamada da atenção
        # self.sublayer[0] é o wrapper de Residual+Norm
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, cos=cos, sin=sin))
            
            # Feed Forward
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
        else:
            x = self.self_attn(x, x, x, mask, cos=cos, sin=sin)
            if self.feed_forward is not None:
                return self.feed_forward(x)
        return x


class RoTHPSimple(TorchBaseModel):
    """
    Versão simplificada do RoTHP usando herança para reduzir código,
    mas mantendo a clareza nos métodos principais.
    """
    def __init__(self, model_config):
        super(RoTHPSimple, self).__init__(model_config)
        
        self.d_model = model_config.hidden_size
        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate
        
        # 1. Módulo RoPE
        self.rotary_emb = RotaryEmbedding(self.d_model // self.n_head)
        
        # 2. Embedding de Tipo (vem do TorchBaseModel) e MLP
        # Recriamos o MLP aqui para ficar claro
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        # 3. Pilha de Camadas (usando nossa versão Rotary)
        self.stack_layers = nn.ModuleList([
            RotaryEncoderLayer(
                self.d_model,
                # Atenção Rotary
                RotaryAttention(self.n_head, self.d_model, self.d_model, self.dropout, output_linear=False),
                feed_forward=self.feed_forward,
                use_residual=False, # RoTHP original não usa residual no bloco externo por padrão
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])
        
        # 4. Cabeçalho de Intensidade
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)
        
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)

    def forward(self, time_seqs, type_seqs, attention_mask):
        # 1. Calcula RoPE
        cos, sin = self.rotary_emb(time_seqs)
        
        # 2. Embedding de Tipo
        enc_output = self.layer_type_emb(type_seqs)
        
        # 3. Passa pelas camadas injetando rotação
        for enc_layer in self.stack_layers:
            enc_output = enc_layer(enc_output, mask=attention_mask, cos=cos, sin=sin)
            
        return enc_output

    def loglike_loss(self, batch):
        """Calcula a perda (Log-Likelihood)"""
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        # Forward (estado oculto h_t)
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        # Intensidade no evento (lambda)
        # lambda(t) = softplus(f(h) + decay * delta_t + base)
        factor_decay = self.factor_intensity_decay[None, ...]
        factor_base = self.factor_intensity_base[None, ...]
        
        intensity_states = (
            factor_decay * time_delta_seqs[:, 1:, None] +
            self.layer_intensity_hidden(enc_out) +
            factor_base
        )
        lambda_at_event = self.softplus(intensity_states)
        
        # Integral da intensidade (via Monte Carlo)
        # 1. Gera tempos amostrais
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])
        
        # 2. Calcula estados nesses tempos
        # Expandimos dimensões para processar amostras em paralelo
        event_states = enc_out[:, :, None, :]
        sample_dt = sample_dtimes[..., None]
        
        intensity_samples = (
            self.factor_intensity_decay[None, None, ...] * sample_dt +
            self.layer_intensity_hidden(event_states) +
            self.factor_intensity_base[None, None, ...]
        )
        lambda_t_sample = self.softplus(intensity_samples)
        
        # 3. Combina Log(lambda) - Integral(lambda)
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event, lambda_t_sample,
            time_delta_seqs[:, 1:], batch_non_pad_mask[:, 1:], type_seqs[:, 1:]
        )
        
        return -(event_ll - non_event_ll).sum(), num_events
