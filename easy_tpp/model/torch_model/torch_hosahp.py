import torch.nn as nn

from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_sahp import SAHP
from easy_tpp.model.torch_model.torch_hothp import (
    HyperbolicRotaryEmbedding,
    HyperbolicMultiHeadAttention,
    HyperbolicEncoderLayer,
)


class HoSAHP(SAHP):
    """Torch implementation of Hyperbolic Rotary Position Embedding-based Self-Attentive Hawkes Process (HoSAHP).

    Same relation to SAHP that HoTHP has to THP: the additive temporal positional
    encoding is dropped and the hyperbolic (HoPE) decay kernel is applied directly
    inside the attention scores. The intensity parameterization of SAHP — the
    exponential state decay mu + (eta - mu) * exp(-gamma * tau) — is kept untouched,
    so only the way the history is *read* (the attention) changes.
    """

    def __init__(self, model_config):
        # bypass SAHP.__init__ (which builds the additive positional encoding and the
        # vanilla attention stack) and set up the SAHP intensity machinery by hand.
        TorchBaseModel.__init__(self, model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # HoPE kernel replaces SAHP's TimeShiftedPositionalEncoding
        self.hope_emb = HyperbolicRotaryEmbedding(self.d_model // self.n_head)

        # kept for parity with SAHP (harmless if unused by state_decay path)
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)  # learnable mark-specific beta

        # hyperbolic attention stack (mirrors HoTHP), no additive positional encoding
        self.stack_layers = nn.ModuleList(
            [HyperbolicEncoderLayer(
                self.d_model,
                HyperbolicMultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                             output_linear=False),
                use_residual=False,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)

        # SAHP intensity parameterization — kept identical (Equations 12-15 of SAHP)
        # mu = GELU(h*W_mu)
        self.mu = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.GELU(),
        )
        # eta = GELU(h*W_eta)
        self.eta = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.GELU(),
        )
        # gamma = Softplus(h*W_gamma)
        self.gamma = nn.Sequential(
            nn.Linear(self.d_model, self.num_event_types, bias=False),
            nn.Softplus(),
        )

    def _normalize_timestamps(self, time_seqs):
        # shift-to-zero, same as HoTHP: keeps relative distances, avoids large abs times
        return time_seqs - time_seqs[:, :1]

    def forward(self, time_seqs, time_delta_seqs, event_seqs, attention_mask):
        """Encode the history with hyperbolic attention.

        Signature kept identical to SAHP.forward so the inherited
        loglike_loss / compute_intensities_at_sample_times keep working.
        time_delta_seqs is accepted for signature compatibility but not used:
        temporal information now enters through the hyperbolic kernel, which
        works off the (normalized) absolute timestamps.

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs (unused).
            event_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.

        Returns:
            tensor: hidden states at event times, [batch_size, seq_len, hidden_size].
        """
        norm_times = self._normalize_timestamps(time_seqs)
        enc_output = self.layer_type_emb(event_seqs)

        thetas = self.hope_emb.thetas
        theta_prime = self.hope_emb.theta_prime

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask,
                time_seqs=norm_times,
                thetas=thetas,
                theta_prime=theta_prime)
            if self.use_norm:
                enc_output = self.norm(enc_output)

        return enc_output
