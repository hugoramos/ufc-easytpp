import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_tpp.model.torch_model.torch_rothp import RoTHP, RotaryEmbedding, apply_rotary_pos_emb
from easy_tpp.model.torch_model.torch_baselayer import RotaryEncoderLayer, RotaryMultiHeadAttention

class IntensityModulatedRotaryEmbedding(RotaryEmbedding):
    """
    IM-RoPE: Intensity-Modulated Rotary Position Embedding.
    A frequência de rotação é modulada pela intensidade do processo.
    """
    def __init__(self, dim, max_freq=10000):
        super().__init__(dim, max_freq)
        # Parâmetro aprendível para controlar a força da modulação
        # Inicializado pequeno para começar próximo do RoPE padrão
        self.alpha = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, t, intensity):
        """
        Args:
            t: [batch, seq_len] (timestamps)
            intensity: [batch, seq_len] (intensidade estimada no tempo t)
        Returns:
            cos, sin: [batch, seq_len, dim]
        """
        # Modulação da Frequência
        # theta_new = theta_base * (1 + alpha * log(1 + lambda))
        # Usamos log(1+x) para estabilidade e suavidade
        
        # [batch, seq_len, 1]
        modulation = 1.0 + self.alpha * torch.log1p(intensity.unsqueeze(-1))
        
        # Expandir t: [batch, seq_len, 1]
        t_expanded = t.unsqueeze(-1)
        
        # Expandir thetas base: [1, 1, dim/2]
        thetas_expanded = self.thetas.view(1, 1, -1)
        
        # Argumento modulado: t * theta * modulation
        # O tempo "corre" mais rápido ou mais devagar dependendo da intensidade
        args = t_expanded * thetas_expanded * modulation
        
        cos_args = torch.cos(args)
        sin_args = torch.sin(args)
        
        cos = torch.repeat_interleave(cos_args, 2, dim=-1)
        sin = torch.repeat_interleave(sin_args, 2, dim=-1)
        
        return cos, sin

class IMRoTHP(RoTHP):
    """
    Intensity-Modulated Rotary Transformer Hawkes Process.
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # Substituir o RoPE padrão pelo IM-RoPE
        self.rotary_emb = IntensityModulatedRotaryEmbedding(self.d_model // self.n_head)
        
        # Camada leve para estimar a intensidade "bruta" a partir do embedding do evento
        # Isso serve de proxy para a intensidade real antes da atenção
        self.intensity_proxy_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softplus() # Intensidade deve ser positiva
        )

    def forward(self, time_seqs, type_seqs, attention_mask):
        # 1. Embedding do Evento
        # [batch, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)
        
        # 2. Estimar Intensidade Proxy (para modular o RoPE)
        # Baseado apenas no tipo do evento (pode ser enriquecido com delta t se quiser)
        # [batch, seq_len]
        intensity_proxy = self.intensity_proxy_layer(type_emb).squeeze(-1)
        
        # 3. Calcular IM-RoPE
        # [batch, seq_len, dim]
        cos, sin = self.rotary_emb(time_seqs, intensity_proxy)

        # 4. Passar pelo Transformer (igual ao RoTHP)
        enc_output = type_emb
        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                cos=cos, 
                sin=sin)

        return enc_output
