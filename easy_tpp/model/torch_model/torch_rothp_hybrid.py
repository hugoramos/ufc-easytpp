import torch
import torch.nn as nn

from easy_tpp.model.torch_model.torch_baselayer import TimePositionalEncoding
from easy_tpp.model.torch_model.torch_rothp import RoTHP

class RoTHPHybrid(RoTHP):
    """
    RoTHP Híbrido: Combina Embeddings de Posição Rotacional (na atenção)
    com Embeddings de Tempo Sinusoidal (somados à entrada).
    
    Esta versão inclui NORMALIZAÇÃO TEMPORAL para tornar o componente
    de embedding absoluto resiliente a grandes deslocamentos de tempo (time shift).
    """

    def __init__(self, model_config):
        """Inicializa o modelo

        Args:
            model_config (EasyTPP.ModelConfig): configuração das especificações do modelo.
        """
        super(RoTHPHybrid, self).__init__(model_config)
        # Inicializa o encoding temporal sinusoidal explícito (que não existe no RoTHP padrão)
        # É a mesma codificação usada no THP original
        self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Executa o modelo

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequências de timestamps absolutos.
            type_seqs (tensor): [batch_size, seq_len], sequências de tipos de evento.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], máscaras de atenção.

        Returns:
            tensor: estados ocultos (hidden states) nos tempos dos eventos.
        """
        # 1. Calcula RoPE (Cos/Sin) para Q e K
        # O RoPE é naturalmente robusto a translações na atenção (pois depende de t_j - t_i),
        # então podemos passar o tempo original ou normalizado.
        # Passar o original mantém a definição estrita do paper.
        cos, sin = self.rotary_emb(time_seqs)

        # 2. Embedding do Tipo de Evento
        enc_output = self.layer_type_emb(type_seqs)
        
        # 3. NORMALIZAÇÃO TEMPORAL (A Correção)
        # O Embedding Absoluto (TimePositionalEncoding) é sensível a shifts globais.
        # Se t=100.000, o seno(100.000) gera valores que o modelo não viu no treino (t=0..100).
        # Solução: Subtraímos o tempo inicial da sequência para zerar o referencial.
        # Isso torna o embedding absoluto "relativo ao início da sequência".
        
        # [batch_size, 1] - Pega o primeiro tempo de cada sequência no batch
        start_times = time_seqs[:, 0].unsqueeze(1)
        
        # [batch_size, seq_len] - Subtrai o início (broadcasting)
        normalized_time = time_seqs - start_times
        
        # 4. ADIÇÃO HÍBRIDA: Injeta Embedding de Tempo Sinusoidal (NORMALIZADO)
        # Agora o modelo recebe informação de tempo consistente (ex: 0.0, 0.5, 1.2...)
        # independente se o relógio original marcava 1000, 1000.5, 1001.2...
        tem_enc = self.layer_temporal_encoding(normalized_time)
        enc_output = enc_output + tem_enc

        # 5. Passa pelas camadas (usando RoPE na atenção)
        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                cos=cos, 
                sin=sin)

        return enc_output
