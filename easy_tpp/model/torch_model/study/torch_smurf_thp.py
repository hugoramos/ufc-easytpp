import torch
import torch.nn as nn
import numpy as np

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_thp import THP


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.d_model = d_model
        # Trainable Fourier features or just sinusoidal
        self.linear_layer = nn.Linear(1, d_model // 2)

    def forward(self, x):
        # x: [batch, ...]
        x_emb = self.linear_layer(x.unsqueeze(-1))
        x_emb = torch.cat([x_emb.sin(), x_emb.cos()], dim=-1)
        return x_emb


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps / sigmas."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sampled weights. We wrap in Parameter but do not train them (usually)
        # or we can train them. Standard NCSN uses fixed weights.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        # x: [batch, seq_len] or [batch, seq_len, 1]
        x_proj = x.unsqueeze(-1) * self.W[None, None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SmurfTHP(THP):
    """
    Torch implementation of SMURF-THP: Score Matching-based Uncertainty Quantification for Transformer Hawkes Process.
    ICML 2023.
    """

    def __init__(self, model_config):
        super(SmurfTHP, self).__init__(model_config)
        
        # SMURF specific configs
        self.sigma_begin = model_config.model_specs.get('sigma_begin', 10.0)
        self.sigma_end = model_config.model_specs.get('sigma_end', 0.01)
        self.num_classes_sigma = model_config.model_specs.get('num_classes_sigma', 50)
        self.sampling_steps = model_config.model_specs.get('sampling_steps', 100)
        self.step_lr = model_config.model_specs.get('step_lr', 2e-5) # Langevin step size
        
        # Sigmas geometric sequence
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.sigma_begin), np.log(self.sigma_end),
                               self.num_classes_sigma))).float().to(self.device)
        self.register_buffer('sigmas', sigmas)
        
        # Score Network
        # Takes [History_Emb; Time_Emb; Sigma_Emb] -> Score (scalar)
        self.time_embed_layer = TimeEmbedding(self.d_model)
        self.sigma_embed_layer = GaussianFourierProjection(self.d_model)
        
        # Conditional Score Network: MLP
        # Input: Hidden (d_model) + Time_Emb (d_model) + Sigma_Emb (d_model) = 3 * d_model
        # Output: 1 (score value)
        self.score_net = nn.Sequential(
            nn.Linear(3 * self.d_model, 2 * self.d_model),
            nn.Softplus(),
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.Softplus(),
            nn.Linear(2 * self.d_model, 1)
        )
        
        # Re-initialize type predictor if needed (THP uses layer_intensity_hidden + softplus for intensity)
        # We just need P(type | history). Standard Softmax over linear projection is enough.
        # THP uses intensity for types too. 
        # In SMURF paper, type prediction is usually a separate classification head trained via CrossEntropy.
        # We can reuse self.layer_intensity_hidden from base THP for logits.

    def get_score(self, history_emb, time_delta, sigma):
        """
        Compute score for given history embedding, time delta candidate, and sigma.
        
        Args:
            history_emb: [batch, seq_len, d_model]
            time_delta: [batch, seq_len]
            sigma: [batch, seq_len] or scalar broadcastable
            
        Returns:
            score: [batch, seq_len]
        """
        # Embed time
        # [batch, seq_len, d_model]
        t_emb = self.time_embed_layer(time_delta)
        
        # Embed Sigma
        if isinstance(sigma, torch.Tensor) and sigma.dim() == 2:
             # Already [batch, seq_len]
             s_emb = self.sigma_embed_layer(sigma)
        else:
             # Broadcast
             # Logic to handle scalar sigma or tensor sigma
             # If sigma is tensor but missing dims, expand
             pass # Simplified for now assuming input match
             s_emb = self.sigma_embed_layer(sigma)
        
        # Concat
        # [batch, seq_len, 3 * d_model]
        inp = torch.cat([history_emb, t_emb, s_emb], dim=-1)
        
        # MLP
        # [batch, seq_len, 1]
        out = self.score_net(inp)
        
        # Convention: Divide output by sigma? (Standard in some implementations)
        # Or just learn the raw score.
        # NCSN usually outputs the score * sigma (predicting the noise z)
        # But here we predict raw score.
        
        return out.squeeze(-1)

    def loglike_loss(self, batch):
        """
        Compute SMURF Loss = Type Loss (NLL) + Time Loss (Score Matching).
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        # 1. Forward Pass to get history embeddings
        # [batch, seq_len, d_model]
        # We mask the last event for prediction as usual?
        # Standard TPP training: predict event i+1 given 0...i
        # forward() in THP returns enc_output at index i (representing h_i).
        # We want to predict t_{i+1} and k_{i+1}.
        
        # History embeddings for steps 0 to N-1
        # Input to forward: time_seqs[:, :-1], ...
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        # Targets
        target_time_deltas = time_delta_seqs[:, 1:] # [batch, seq_len]
        target_types = type_seqs[:, 1:] # [batch, seq_len]
        mask = batch_non_pad_mask[:, 1:] # [batch, seq_len]
        
        # --- Type Loss ---
        # Simple Cross Entropy on logits
        # [batch, seq_len, num_event_types]
        type_logits = self.layer_intensity_hidden(enc_out)
        
        # Masked Cross Entropy works best with flatten
        active_logits = type_logits[mask.bool()]
        active_labels = target_types[mask.bool()].long()
        
        type_loss = nn.CrossEntropyLoss()(active_logits, active_labels)
        
        # --- Time Loss (Denoising Score Matching) ---
        # Choose random sigma for each element in batch/seq
        # [batch, seq_len]
        rand_indices = torch.randint(0, self.num_classes_sigma, target_time_deltas.shape, device=self.device)
        used_sigmas = self.sigmas[rand_indices] # [batch, seq_len]
        
        # Sample noise z ~ N(0, 1)
        z = torch.randn_like(target_time_deltas)
        
        # Perturbed data: x_tilde = x + sigma * z
        # We might need to ensure x_tilde > 0? The paper discusses this.
        # Often doing score matching in log domain (log time) helps with positivity.
        # Let's assume standard domain for simplicity as per quick read, 
        # but apply ReLU or Softplus in sampling if needed. 
        # Ideally we train to score match on the positive real line.
        # Simply perturbing:
        perturbed_time_deltas = target_time_deltas + used_sigmas * z
        
        # Compute Score
        # s_theta(x_tilde, h, sigma)
        predicted_score = self.get_score(enc_out, perturbed_time_deltas, used_sigmas)
        
        # Target score: - (x_tilde - x) / sigma^2 = - (sigma * z) / sigma^2 = - z / sigma
        target_score = - z / used_sigmas
        
        # DSM Loss = 0.5 * || score - target ||^2 * sigma^2 (often weighted like this)
        # or just MSE.
        # Paper usually uses: expected || score - target ||^2
        # We also need to optimize the conditioning on sigma if score net takes sigma.
        # Wait, usually Noise Conditional Score Networks (NCSN) take sigma as input too!
        # My score_net defined above does NOT take sigma.
        # I should add sigma embedding to score_net.
        
        # Let's correct this in a separate update if needed, but for "style" consistency with basic implementation:
        # If the model is Annealed Langevin, it MUST condition on sigma.
        # If it's just Denoising Score Matching with fixed sigma, it doesn't.
        # "SMURF-THP" typically implies NCSN (Annealed).
        # I will accept a slight simplification if complex, but let's try to add sigma injection.
        
        # Simplified Loss (assuming sigma is implicitly handled or just summation):
        # We calculate squared error.
        score_diff = predicted_score - target_score
        time_loss_elementwise = 0.5 * (score_diff ** 2) * (used_sigmas ** 2) # Weighting?
        # Proper weighting for NCSN is typically * sigma^2 to keep loss magnitude balanced.
        
        time_loss = (time_loss_elementwise * mask).sum() / (mask.sum() + 1e-9)
        
        # Total Loss
        total_loss = type_loss + time_loss
        
        return total_loss, mask.sum()

    def predict_one_step_at_every_event(self, batch):
        """
        Predict next using Langevin Dynamics.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        
        # 1. Forward History
        # We use the whole sequence to predict the NEXT event (one step after the last)
        # Or do we predict for ALL steps? The method name implies "at every event".
        # Let's do it for all steps 0..N-1 to predict 1..N
        
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])
        
        # 2. Type Prediction
        type_logits = self.layer_intensity_hidden(enc_out)
        types_pred = torch.argmax(type_logits, dim=-1) # [batch, seq_len]
        
        # 3. Time Prediction (Langevin Dynamics)
        # We need to sample for every position in the batch/seq.
        # Initial random samples
        # [batch, seq_len]
        x = torch.rand_like(time_delta_seqs[:, 1:]) * 5.0 # Random initialization
        
        # Annealed Langevin Dynamics
        # Iterate through sigmas from large to small
        for sigma in self.sigmas:
            alpha_i = self.step_lr * (sigma / self.sigmas[-1]) ** 2 # Step size schedule
            
            for t in range(self.sampling_steps):
                z = torch.randn_like(x)
                
                # Get score
                # Condition on current sigma
                sigma_in = torch.full_like(x, sigma)
                score = self.get_score(enc_out, x, sigma_in)
                
                x = x + 0.5 * alpha_i * score + torch.sqrt(alpha_i) * z
                
        # Enforce positive
        dtimes_pred = x.abs() 
        
        return dtimes_pred, types_pred
