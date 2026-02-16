import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_tpp.model.torch_model.torch_rothp import RoTHP

class RoTHPDecay(RoTHP):
    """
    RoTHP with Exponential Decay (RoTHP-Decay).
    Combines:
    1. Rotary Position Embeddings (from RoTHP) for translation invariance.
    2. Exponential Decay Intensity (from THP-ExpDecay) for inductive bias.
    
    lambda(t) = softplus(f(h) * exp(-delta * dt) + beta)
    """
    def __init__(self, model_config):
        super(RoTHPDecay, self).__init__(model_config)
        
        # Re-initialize decay parameter to encourage visible decay
        # Same as THPExpDecay
        nn.init.uniform_(self.factor_intensity_decay, 1.0, 3.0)

    def _compute_intensity_with_exp_decay(self, hidden_states, dtimes):
        """Core intensity computation with exponential decay."""
        # Ensure decay rate is positive via softplus
        decay_rate = F.softplus(self.factor_intensity_decay)
        
        # Project hidden states to event-type space
        hidden_intensity = self.layer_intensity_hidden(hidden_states)
        
        # Base intensity
        base = self.factor_intensity_base
        
        # Exponential decay
        intensity_states = hidden_intensity * torch.exp(-decay_rate * dtimes) + base
        
        return intensity_states

    def loglike_loss(self, batch):
        """Compute the loglike loss with exponential decay."""
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        # 1. Forward (RoPE Encoder)
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        # 2. Intensity with Exp Decay
        intensity_states = self._compute_intensity_with_exp_decay(
            enc_out,
            time_delta_seqs[:, 1:, None]
        )
        
        lambda_at_event = self.softplus(intensity_states)
        
        # 3. Integral (MC Sampling)
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])
        state_t_sample = self.compute_states_at_sample_times(enc_out, sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:])
            
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute hidden states at sampled times."""
        # [batch, seq, 1, hidden]
        event_states = event_states[:, :, None, :]
        # [batch, seq, samples, 1]
        sample_dtimes = sample_dtimes[..., None]
        
        intensity_states = self._compute_intensity_with_exp_decay(
            event_states,
            sample_dtimes
        )
        return intensity_states
