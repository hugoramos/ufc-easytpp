"""THP with Exponential Decay (THP-ExpDecay).

Replaces the linear inter-event decay of the original THP with an exponential decay,
inspired by the continuous-time LSTM mechanism of the Neural Hawkes Process (NHP).

Original THP (Zuo et al., ICML 2020):
    λ(t) = softplus(α · Δt + f(h) + β)           [linear decay]

THP-ExpDecay (proposed):
    λ(t) = softplus(f(h) · exp(-δ · Δt) + β)     [exponential decay]

where:
    - f(h) = linear projection of transformer hidden states at event times
    - δ = softplus(learned_decay_rate) to ensure positivity
    - Δt = time since last event
    - β = learned base intensity per event type

Motivation:
    The THP achieves superior NLL compared to NHP (better density estimation),
    but underperforms on point prediction metrics (Accuracy, RMSE).
    Analysis shows this gap is caused by THP's linear decay approximation
    between events, which poorly models the intensity function's shape.
    NHP's exponential decay naturally bounds the intensity and provides
    smooth, monotonic decay -- properties essential for accurate thinning-based
    prediction.

    THP-ExpDecay combines:
    - Transformer's parallel processing and attention mechanism (from THP)
    - Exponential inter-event decay (from NHP)

Architecture:
    - Inherits ALL Transformer components from THP (encoder, attention, embeddings)
    - Only overrides the inter-event intensity computation (2 methods)
    - Same number of parameters as THP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from easy_tpp.model.torch_model.torch_thp import THP


class THPExpDecay(THP):
    """THP with Exponential Decay between events.

    Inherits the full Transformer architecture from THP and only replaces
    the linear inter-event decay with an exponential decay mechanism.
    """

    def __init__(self, model_config):
        super(THPExpDecay, self).__init__(model_config)

        # Re-initialize decay parameter with positive-friendly init
        # Increased range (1.0 to 3.0) to encourage faster, visible exponential decay
        nn.init.uniform_(self.factor_intensity_decay, 1.0, 3.0)

    def _compute_intensity_with_exp_decay(self, hidden_states, dtimes):
        """Core intensity computation with exponential decay.

        Args:
            hidden_states: transformer output, [..., hidden_size]
            dtimes: time deltas, [..., 1] or [...] (will be unsqueezed)

        Returns:
            intensity states: [..., num_event_types]
        """
        # Ensure decay rate is positive via softplus
        # [1, ..., num_event_types]
        decay_rate = F.softplus(self.factor_intensity_decay)

        # Project hidden states to event-type space
        # [..., num_event_types]
        hidden_intensity = self.layer_intensity_hidden(hidden_states)

        # Base intensity
        base = self.factor_intensity_base

        # Exponential decay: f(h) * exp(-δ * Δt) + β
        # This naturally bounds the intensity and provides smooth monotonic decay
        intensity_states = hidden_intensity * torch.exp(-decay_rate * dtimes) + base

        return intensity_states

    def loglike_loss(self, batch):
        """Compute the loglike loss with exponential decay.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])

        # [batch_size, seq_len, num_event_types]
        # Exponential decay instead of linear
        intensity_states = self._compute_intensity_with_exp_decay(
            enc_out,
            time_delta_seqs[:, 1:, None]  # [batch_size, seq_len, 1]
        )

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:])

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times with exponential decay.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: intensity states at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = self._compute_intensity_with_exp_decay(
            event_states,
            sample_dtimes
        )

        return intensity_states
