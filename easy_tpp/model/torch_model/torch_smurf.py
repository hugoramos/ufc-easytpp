import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_tpp.model.torch_model.torch_thp import THP

def get_sm_loss(t_var, score, non_pad_mask):
    """Calcula Hyvarinen Score Matching Loss: tr(Hessian) + 0.5*||score||^2"""
    # Gradiente do score em relação a t (divergência do score)
    # create_graph=True é necessário para calcular derivadas de segunda ordem durante backprop
    grad_t = torch.autograd.grad(score.sum(), t_var, create_graph=True, retain_graph=True)[0]
    
    loss = grad_t + 0.5 * (score ** 2)
    
    if non_pad_mask is not None:
        loss = loss * non_pad_mask.unsqueeze(-1)
        
    return loss.sum()

class IntensityEncode(nn.Module):
    """Módulo que modela a intensidade parametrizada para Score Matching"""
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.affect_layer = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.Tanh()
        )
        self.base_layer = nn.Sequential(
            nn.Linear(d_model, d_inner)
        )
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_inner, 1),
            nn.Softplus(beta=1.0)
        )
        
    def get_score(self, enc_output, t):
        # enc_output: [B, L, D]
        # t: [B, L, 1]
        
        # Garante que t requer gradiente para autograd
        if not t.requires_grad:
            t.requires_grad_(True)
            
        affect = self.affect_layer(enc_output)
        base = self.base_layer(enc_output)
        
        # Formula do Github: tanh(affect * t + base)
        hidden = torch.tanh(affect * t + base)
        intensity = self.intensity_layer(hidden) # [B, L, 1]
        
        # Score = d/dt (log lambda) - lambda
        # Evitar log(0)
        log_intensity = (intensity + 1e-10).log()
        
        grad_log_intensity = torch.autograd.grad(log_intensity.sum(), t, create_graph=True, retain_graph=True)[0]
        
        score = grad_log_intensity - intensity
        return score, intensity

class SmurfTHP(THP):
    """Implementação do SMURF-THP adaptada para EasyTPP.
    Reference: SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process (ICML 2023)
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        # Substitui a camada de intensidade padrão do THP pelo Score Module
        # d_inner é geralmente 2x d_model no paper/código original
        d_inner = model_config.hidden_size * 2
        self.score_net = IntensityEncode(model_config.hidden_size, d_inner)
        
        # Preditor de tipo dependente do tempo (como no paper)
        # Input: hidden state + time gap
        self.type_classifier_t = nn.Sequential(
            nn.Linear(model_config.hidden_size + 1, model_config.hidden_size),
            nn.ReLU(),
            nn.Linear(model_config.hidden_size, self.num_event_types)
        )

    def loglike_loss(self, batch):
        """Calcula a loss total (Score Matching + Cross Entropy).
        Nota: O nome do método é mantido como loglike_loss para compatibilidade com o runner,
        mas retorna a loss de Score Matching.
        """
        # Batch unpacking
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        # 1. Forward Transformer (Encoder)
        # Inputs: t_0 ... t_{N-1}
        # [B, L, D]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        # 2. Score Matching Loss (Tempo)
        # Targets: Delta t_1 ... t_N
        target_deltas = time_delta_seqs[:, 1:].unsqueeze(-1).clone().detach() # [B, L, 1]
        target_deltas.requires_grad_(True)
        target_mask = batch_non_pad_mask[:, 1:] # [B, L]
        
        # Calcula Score
        score, _ = self.score_net.get_score(enc_out, target_deltas)
        
        # Calcula Loss de SM
        sm_loss = get_sm_loss(target_deltas, score, target_mask)
        
        # 3. Type Prediction Loss (Cross Entropy)
        # Usa o tempo real (target_deltas) como input para ajudar a prever o tipo
        # Concatenar hidden state + tempo futuro (teacher forcing)
        type_input = torch.cat([enc_out, target_deltas], dim=-1)
        type_logits = self.type_classifier_t(type_input)
        
        type_loss = F.cross_entropy(type_logits.transpose(1, 2), type_seqs[:, 1:], reduction='none')
        type_loss = (type_loss * target_mask).sum()
        
        # Loss total (pode adicionar pesos lambda se necessário)
        total_loss = sm_loss + type_loss
        num_events = target_mask.sum()
        
        return total_loss, num_events

    def predict_one_step_at_every_event(self, batch):
        """Predição usando Langevin Dynamics Sampling."""
        # Batch unpacking
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        with torch.no_grad():
            enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
            
        # Inicialização aleatória para Langevin
        B, L, _ = enc_out.shape
        # Inicia com 1.0 (média aproximada dos deltas normalizados)
        t_sample = torch.ones(B, L, 1, device=enc_out.device, requires_grad=True)
        
        # Langevin loop parameters
        n_steps = 20 
        step_size = 0.1
        
        for _ in range(n_steps):
            # Score = grad_log_p
            # Langevin: t_new = t + step * score + noise
            # Precisamos re-habilitar gradientes apenas para o cálculo do score
            with torch.enable_grad():
                score, _ = self.score_net.get_score(enc_out.detach(), t_sample)
            
            noise = torch.randn_like(t_sample) * (2 * step_size)**0.5
            t_sample = t_sample.detach() + step_size * score.detach() + noise
            t_sample = F.relu(t_sample) # Tempo deve ser positivo
            t_sample.requires_grad_(True)
            
        # Predição final de tempo
        dtimes_pred = t_sample.squeeze(-1).detach()
        
        # Predição de tipo usando o tempo estimado
        type_input = torch.cat([enc_out, t_sample.detach()], dim=-1)
        type_logits = self.type_classifier_t(type_input)
        types_pred = torch.argmax(type_logits, dim=-1)
        
        return dtimes_pred, types_pred

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Calcula intensidade para visualização."""
        attention_mask = kwargs.get('attention_mask')
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)
        
        # Expandir para amostras
        # enc_out: [B, L, D] -> [B, L, K, D]
        K = sample_dtimes.shape[-1]
        enc_expanded = enc_out.unsqueeze(2).expand(-1, -1, K, -1)
        
        # sample_dtimes: [B, L, K]
        t_expanded = sample_dtimes.unsqueeze(-1) # [B, L, K, 1]
        t_expanded.requires_grad_(True)
        
        # Score net retorna score e intensidade
        _, intensity = self.score_net.get_score(enc_expanded, t_expanded)
        
        # intensity: [B, L, K, 1]
        # Replicar para todos os tipos (SMURF univariado no tempo)
        intensity_total = intensity.expand(-1, -1, -1, self.num_event_types)
        
        return intensity_total
