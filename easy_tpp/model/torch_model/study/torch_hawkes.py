import torch
import torch.nn as nn
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

class Hawkes(TorchBaseModel):
    """
    Classical Hawkes Process Model
    
    Formula: lambda_m(t) = mu_m + sum of past influences
    
    Parameters:
        - mu: base intensity (how often events happen naturally)
        - alpha: influence strength (how much one event triggers another)
        - beta: decay rate (how fast the influence fades)
    """
    
    def __init__(self, model_config):
        super(Hawkes, self).__init__(model_config)
        self.num_types = model_config.num_event_types
        
        # Initialize learnable parameters
        # mu: base intensity for each event type [num_types]
        self.mu = nn.Parameter(torch.rand(self.num_types) + 0.1) # para evitar intensidade inercial zero (muito? pouco?)
        
        # alpha: how much event type j influences type m [num_types, num_types]
        self.alpha = nn.Parameter(torch.rand(self.num_types, self.num_types))
        
        # beta: decay rate of influence from j to m [num_types, num_types]
        self.beta = nn.Parameter(torch.rand(self.num_types, self.num_types) + 0.1) # para evitar decaimento zero

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Not used in classical Hawkes, kept for compatibility"""
        return None

    def loglike_loss(self, batch):
        """
        Calculate the loss for training
        Returns: (loss, number_of_events)
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        
        loss = self._compute_loss(time_delta_seqs, type_seqs, batch_non_pad_mask)
        num_events = batch_non_pad_mask[:, 1:].sum()
        
        return loss, num_events

    def _compute_loss(self, time_deltas, types, mask):
        """
        Compute negative log-likelihood
        
        Log-likelihood = sum(log(intensity at events)) - integral(intensity over time)
        """
        batch_size, seq_len = time_deltas.size()
        
        # Prepare parameters for batch processing - [batch, M] or [batch, M, M]
        mu = self.mu.unsqueeze(0)  # [1, M]
        alpha = self.alpha.unsqueeze(0)  # [1, M, M]
        beta = self.beta.unsqueeze(0)  # [1, M, M]
        
        # Accumulators for log-likelihood components
        log_intensity_sum = 0.0
        integral_sum = 0.0
        
        # Track influence from past events (recursive state)
        influence = torch.zeros(batch_size, self.num_types, self.num_types, device=self.device)
        
        # Process each event in the sequence
        for step in range(seq_len):
            
            # Update influence from previous event
            if step > 0:
                time_diff = time_deltas[:, step]  # [batch] - use pre-calculated delta
                prev_event_type = types[:, step-1].long()  # [batch]
                
                # One-hot encoding for previous event type: [batch, M]
                event_indicator = torch.nn.functional.one_hot(
                    prev_event_type, 
                    num_classes=self.num_types
                ).float()
                
                # Decay factor: [batch, M, M]
                decay_factor = torch.exp(-beta * time_diff.view(-1, 1, 1))
                
                # Update influence with decay
                new_influence = alpha * event_indicator.unsqueeze(1)  # [batch, M, M]
                influence = (influence + new_influence) * decay_factor
                
                # Calculate integral term (compensator)
                mu_integral = mu * time_diff.unsqueeze(1)  # [batch, M]
                
                influence_before_decay = influence / decay_factor
                influence_integral = (influence_before_decay * (1 - decay_factor) / beta).sum(dim=2)
                
                total_integral = (mu_integral + influence_integral).sum(dim=1)  # [batch]
                integral_sum = integral_sum + (total_integral * mask[:, step]).sum()
            
            # Calculate intensity at current event: [batch, M]
            current_intensity = mu + influence.sum(dim=2)
            
            # Get intensity for the actual event type that occurred
            current_event_type = types[:, step].long()
            event_intensity = current_intensity.gather(1, current_event_type.unsqueeze(1)).squeeze(1)
            
            # Ensure numerical stability (avoid log(0))
            event_intensity = torch.clamp(event_intensity, min=1e-8)
            
            # Add to log-likelihood (only for non-padded events)
            log_intensity_sum = log_intensity_sum + (torch.log(event_intensity) * mask[:, step]).sum()
        
        # Negative log-likelihood
        return -(log_intensity_sum - integral_sum)

    def compute_residuals(self, batch):
        """
        Compute residuals for model evaluation
        
        Residuals transform the irregular event times into a unit-rate Poisson process
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        batch_size, seq_len = time_seqs.size()
        
        residuals = torch.zeros_like(time_seqs)
        
        # Prepare parameters - [batch, M] or [batch, M, M]
        mu = self.mu.unsqueeze(0)  # [1, M]
        alpha = self.alpha.unsqueeze(0)  # [1, M, M]
        beta = self.beta.unsqueeze(0)  # [1, M, M]
        
        influence = torch.zeros(batch_size, self.num_types, self.num_types, device=self.device)
        
        for step in range(1, seq_len):
            time_diff = time_delta_seqs[:, step]  # Use pre-calculated delta
            prev_event_type = type_seqs[:, step-1].long()
            
            # One-hot encoding: [batch, M]
            event_indicator = torch.nn.functional.one_hot(
                prev_event_type, 
                num_classes=self.num_types
            ).float()
            
            decay_factor = torch.exp(-beta * time_diff.view(-1, 1, 1))
            new_influence = alpha * event_indicator.unsqueeze(1)
            
            # Calculate integral for this time interval
            influence_with_jump = influence + new_influence
            
            mu_part = mu * time_diff.unsqueeze(1)
            influence_part = (influence_with_jump * (1 - decay_factor) / beta).sum(dim=2)
            
            interval_integral = (mu_part + influence_part).sum(dim=1)
            residuals[:, step] = interval_integral
            
            # Update influence for next step
            influence = influence_with_jump * decay_factor
        
        # Cumulative sum gives transformed timestamps
        tau_seqs = torch.cumsum(residuals, dim=1)
        
        return tau_seqs, residuals
