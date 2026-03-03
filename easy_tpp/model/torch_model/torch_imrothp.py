import torch
import torch.nn as nn
import torch.nn.functional as F
# Corrigindo imports: As classes Rotary estão em torch_rothp, não em baselayer
from easy_tpp.model.torch_model.torch_rothp import RoTHP, RotaryEmbedding, apply_rotary_pos_emb, RotaryEncoderLayer, RotaryMultiHeadAttention

class IntensityModulatedRotaryEmbedding(RotaryEmbedding):
    """
    IM-RoPE: Intensity-Modulated Rotary Position Embedding.
    A frequência de rotação é modulada, ponderada pela intensidade do processo.
    """
    def __init__(self, dim, max_freq=10000):
        super().__init__(dim, max_freq)
        self.alpha = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, time_seqs, intensity):
        # time_seqs original: [batch, seq_len]
        # time_seqs_expanded: [batch, seq_len, 1]
        time_seqs_expanded = time_seqs.unsqueeze(-1)
        
        # self.thetas: [dim/2] (vetor com as frequencias)
        # thetas_expanded: [1, 1, dim/2]
        thetas_expanded = self.thetas.view(1, 1, -1)
        
        # args = time_seqs * theta (tempo * frequência)
        # args: [batch, seq_len, dim/2]
        args = time_seqs_expanded * thetas_expanded

        # Argumento modulado: t * theta * modulation
        # O tempo "corre" mais rápido ou mais devagar dependendo da intensidade
        modulation = 1.0 + self.alpha * torch.log1p(intensity.unsqueeze(-1)) # quando for 0, fica 0
        args = time_seqs_expanded * thetas_expanded * modulation
        
        cos_args = torch.cos(args)
        sin_args = torch.sin(args)
        
        return cos_args, sin_args

class IMRoTHP(RoTHP):
    """
    Intensity-Modulated Rotary Transformer Hawkes Process.
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # mudei do RoPE padrão pelo IM-RoPE
        self.rotary_emb = IntensityModulatedRotaryEmbedding(self.d_model // self.n_head)
        
        self.intensity_proxy_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softplus() # pra ser positiva
        )

    def forward(self, time_seqs, type_seqs, attention_mask):
        # [batch, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)
        
        # [batch, seq_len]
        # tentativa de pre calcular intensidade (apenas tipo)
        intensity_proxy = self.intensity_proxy_layer(type_emb).squeeze(-1)
        
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
