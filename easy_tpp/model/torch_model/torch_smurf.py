import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_tpp.model.torch_model.torch_thp import THP

def get_sm_loss(t_var, score, non_pad_mask):
    """Calcula Hyvarinen Score Matching Loss"""
    if not t_var.is_contiguous(): t_var = t_var.contiguous()
    if not score.is_contiguous(): score = score.contiguous()
    
    grad_t = torch.autograd.grad(
        score.sum(), t_var, 
        create_graph=True, 
        retain_graph=True,
        only_inputs=True
    )[0]
    
    loss = grad_t + 0.5 * (score ** 2)
    
    if non_pad_mask is not None:
        loss = loss * non_pad_mask.unsqueeze(-1)
        
    return loss.sum()

class IntensityEncode(nn.Module):
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
        if not t.requires_grad:
            t.requires_grad_(True)
            
        enc_output = enc_output.contiguous()
        t = t.contiguous()
            
        affect = self.affect_layer(enc_output)
        base = self.base_layer(enc_output)
        
        hidden = torch.tanh(affect * t + base)
        intensity = self.intensity_layer(hidden)
        
        intensity = torch.clamp(intensity, min=1e-5, max=1e5)
        log_intensity = intensity.log()
        
        grad_log_intensity = torch.autograd.grad(
            log_intensity.sum(), t, 
            create_graph=True, 
            retain_graph=True,
            only_inputs=True
        )[0]
        
        if torch.isnan(grad_log_intensity).any():
            grad_log_intensity = torch.nan_to_num(grad_log_intensity, nan=0.0)
            
        score = grad_log_intensity - intensity
        return score, intensity

class SmurfTHP(THP):
    def __init__(self, model_config):
        super().__init__(model_config)
        d_inner = model_config.hidden_size * 2
        self.score_net = IntensityEncode(model_config.hidden_size, d_inner)
        
        self.type_classifier_t = nn.Sequential(
            nn.Linear(model_config.hidden_size + 1, model_config.hidden_size),
            nn.ReLU(),
            nn.Linear(model_config.hidden_size, self.num_event_types)
        )
        # Pad token ID deve vir do config
        self.pad_token_id = getattr(model_config, 'pad_token_id', self.num_event_types)

    def loglike_loss(self, batch):
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        target_deltas = time_delta_seqs[:, 1:].unsqueeze(-1).clone().detach()
        target_deltas.requires_grad_(True)
        target_mask = batch_non_pad_mask[:, 1:]
        
        score, _ = self.score_net.get_score(enc_out, target_deltas)
        sm_loss = get_sm_loss(target_deltas, score, target_mask)
        
        type_input = torch.cat([enc_out, target_deltas.detach()], dim=-1)
        type_logits = self.type_classifier_t(type_input)
        
        # Correção: ignore_index para evitar erro com token de padding
        type_loss = F.cross_entropy(
            type_logits.transpose(1, 2), 
            type_seqs[:, 1:], 
            reduction='none',
            ignore_index=self.pad_token_id
        )
        type_loss = (type_loss * target_mask).sum()
        
        total_loss = sm_loss + type_loss
        num_events = target_mask.sum() + 1e-9
        
        return total_loss, num_events

    def predict_one_step_at_every_event(self, batch):
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        with torch.no_grad():
            enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
            
        B, L, _ = enc_out.shape
        t_sample = torch.ones(B, L, 1, device=enc_out.device, requires_grad=True)
        
        n_steps = 20 
        step_size = 0.1
        
        for _ in range(n_steps):
            with torch.enable_grad():
                score, _ = self.score_net.get_score(enc_out.detach(), t_sample)
            
            noise = torch.randn_like(t_sample) * (2 * step_size)**0.5
            t_sample = t_sample.detach() + step_size * score.detach() + noise
            t_sample = F.relu(t_sample)
            t_sample.requires_grad_(True)
            
        dtimes_pred = t_sample.squeeze(-1).detach()
        
        type_input = torch.cat([enc_out, t_sample.detach()], dim=-1)
        type_logits = self.type_classifier_t(type_input)
        types_pred = torch.argmax(type_logits, dim=-1)
        
        return dtimes_pred, types_pred

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        attention_mask = kwargs.get('attention_mask')
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)
        
        K = sample_dtimes.shape[-1]
        enc_expanded = enc_out.unsqueeze(2).expand(-1, -1, K, -1)
        t_expanded = sample_dtimes.unsqueeze(-1)
        t_expanded.requires_grad_(True)
        
        _, intensity = self.score_net.get_score(enc_expanded, t_expanded)
        intensity_total = intensity.expand(-1, -1, -1, self.num_event_types)
        return intensity_total
