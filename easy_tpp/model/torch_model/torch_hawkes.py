import torch
import torch.nn as nn
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

class Hawkes(TorchBaseModel):
    def __init__(self, model_config):
        super(Hawkes, self).__init__(model_config)
        self.num_types = model_config.num_event_types
        
        self.mu_raw = nn.Parameter(torch.randn(self.num_types) + 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(self.num_types, self.num_types))
        self.beta_raw = nn.Parameter(torch.randn(self.num_types, self.num_types))

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.mu_raw)
    
    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.alpha_raw)
    
    @property
    def beta(self):
        return torch.nn.functional.softplus(self.beta_raw)

    def forward(self, time_seqs, type_seqs, attention_mask):
        # não é utilizado, mas a interface precisa de um retorno
        return None

    def loglike_loss(self, batch):
        _, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        
        loss = self._compute_loss(time_delta_seqs, type_seqs, batch_non_pad_mask)
        num_events = batch_non_pad_mask[:, 1:].sum()
        
        return loss, num_events

    def _compute_loss(self, time_deltas, types, mask):
        batch_size, seq_len = time_deltas.size()
        
        mu = self.mu.unsqueeze(0)  # [1, M]
        alpha = self.alpha.unsqueeze(0)  # [1, M, M]
        beta = self.beta.unsqueeze(0)  # [1, M, M]
        
        log_intensity_sum = 0.0
        integral_sum = 0.0
        
        # memória de influência: [batch, M, M]
        influence = torch.zeros(batch_size, self.num_types, self.num_types, device=self.device)

        for step in range(seq_len):
            # atualiza a influência a partir do evento anterior
            if step > 0:
                time_diff = time_deltas[:, step]                
                time_diff = torch.clamp(time_diff, min=1e-6) # evitar log(0)
                
                prev_event_type = types[:, step-1].long()
                
                # one-hot: [batch, M] (utilizado no multivariado)
                event_indicator = torch.nn.functional.one_hot(
                    prev_event_type, 
                    num_classes=self.num_types
                ).float()
                
                # fator de decaimento: [batch, M, M]
                # exp(-beta * dt)
                decay_factor = torch.exp(-beta * time_diff.view(-1, 1, 1))
                
                ####### ver com o cesar se isso tá fazendo sentido
                # nova influência adicionada pelo evento anterior: alpha * indicator

                new_influence = alpha * event_indicator.unsqueeze(1)
                
                # atualiza a influência total
                influence = (influence + new_influence) * decay_factor 
                #######

                # calcula a integral do mi (mi * delta t)
                # ∫ μ dt = μ × Δt
                mu_integral = mu * time_diff.unsqueeze(1)
                

                # calcula a integral da influencia (influence * (1 - decay_factor) / beta) (influencia decaindo)
                # ∫ α·exp(-β·t) dt = α/β · (1 - exp(-β·Δt))
                influence_start = influence / decay_factor
                influence_integral = (influence_start * (1 - decay_factor) / beta).sum(dim=2)
                
                total_integral = (mu_integral + influence_integral).sum(dim=1)
                
                # adiciona ao total da integral (apenas para eventos válidos)
                integral_sum = integral_sum + (total_integral * mask[:, step]).sum()
            
            # calcula a intensidade no tempo do evento atual
            # lambda(t) = mu + sum(influence)
            current_intensity = mu + influence.sum(dim=2)
            
            # seleciona a intensidade para o tipo de evento que ocorreu
            current_event_type = types[:, step].long()
            event_intensity = current_intensity.gather(1, current_event_type.unsqueeze(1)).squeeze(1) # no multivariado
            
            # estabibilidade numérica 
            event_intensity = torch.clamp(event_intensity, min=1e-8)
            
            # adiciona log(lambda) à perda (apenas para eventos válidos)
            log_intensity_sum = log_intensity_sum + (torch.log(event_intensity) * mask[:, step]).sum()
        
        # loglikelihood negativa
        return -(log_intensity_sum - integral_sum)
