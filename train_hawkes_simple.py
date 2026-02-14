import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easy_tpp.model.torch_model.torch_hawkes import Hawkes

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    def __init__(self):
        self.num_event_types = 1
        self.num_event_types_pad = 1
        self.gpu = -1
        self.hidden_size = 64 
        self.pad_token_id = -1
        self.loss_integral_num_sample_per_step = 0
        self.use_mc_samples = False
        self.thinning = None
        self.good_init = True # Start close to truth to verify stability

# ============================================================================
# 2. ROBUST DATA GENERATOR (Ogata's Thinning Algorithm)
# ============================================================================
def simulate_hawkes_clean(mu, alpha, beta, num_seqs, max_time=100.0, seed=42):
    """
    Generates Hawkes process sequences with strictly increasing times.
    No duplicates, no artifacts.
    """
    np.random.seed(seed)
    print(f"\n🎲 Generating {num_seqs} sequences (Ogata's Thinning)...")
    print(f"   Params: mu={mu}, alpha={alpha}, beta={beta}")
    
    seqs = []
    
    for _ in range(num_seqs):
        timestamps = []
        t = 0.0
        
        # History of events for intensity calculation
        history = []
        
        while t < max_time:
            # Upper bound for intensity (lambda_bar)
            # Current intensity at time t (decays from last event)
            if not history:
                lambda_bar = mu
            else:
                # Calculate exact intensity at current t
                # sum ( alpha * exp(-beta * (t - ti)) )
                recursive_sum = sum([np.exp(-beta * (t - ti)) for ti in history])
                lambda_t = mu + alpha * recursive_sum
                
                # Use current intensity as upper bound (since it only decays until next event)
                lambda_bar = lambda_t 
            
            # 1. Generate candidate inter-arrival time from homogeneous Poisson(lambda_bar)
            u = np.random.uniform(0, 1)
            w = -np.log(u) / lambda_bar
            t += w
            
            if t >= max_time:
                break
                
            # 2. Acceptance/Rejection (Thinning)
            # Calculate actual intensity at candidate time t
            recursive_sum = sum([np.exp(-beta * (t - ti)) for ti in history])
            lambda_actual = mu + alpha * recursive_sum
            
            d = np.random.uniform(0, 1)
            if d * lambda_bar <= lambda_actual:
                # Accept
                timestamps.append(t)
                history.append(t)
        
        seqs.append(timestamps)
    
    return seqs

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
def prepare_batch(raw_seqs):
    num_seqs = len(raw_seqs)
    max_len = max(len(s) for s in raw_seqs)
    
    # We need a start token at 0.0 for the model to calculate the first interval
    # The generated events are t1, t2, ... > 0
    # Model expects: [0.0, t1, t2, ..., tn, pad, pad]
    
    model_max_len = max_len + 1 
    
    time_seqs = torch.zeros(num_seqs, model_max_len)
    time_deltas = torch.zeros(num_seqs, model_max_len)
    type_seqs = torch.zeros(num_seqs, model_max_len).long()
    mask = torch.zeros(num_seqs, model_max_len)
    
    for i, seq in enumerate(raw_seqs):
        # Add 0.0 at start
        full_seq = [0.0] + seq 
        l = len(full_seq)
        
        # To Tensor
        t_tensor = torch.tensor(full_seq, dtype=torch.float32)
        
        # Fill Time
        time_seqs[i, :l] = t_tensor
        # Fill Mask (1 for events including start 0.0)
        mask[i, :l] = 1.0
        
        # Fill Delta
        # delta[0] = 0
        # delta[k] = time[k] - time[k-1]
        if l > 1:
            time_deltas[i, 1:l] = t_tensor[1:] - t_tensor[:-1]
            
        # Padding (repeat last value)
        if l < model_max_len:
            time_seqs[i, l:] = full_seq[-1]
            # Deltas and mask remain 0 for padding
            
    return (time_seqs, time_deltas, type_seqs, mask, None)

# ============================================================================
# 4. MAIN LOOP
# ============================================================================
def main():
    # --- Settings ---
    TRUE_MU = 0.5
    TRUE_ALPHA = 0.3
    TRUE_BETA = 1.0
    
    # --- Generate Data ---
    train_raw = simulate_hawkes_clean(TRUE_MU, TRUE_ALPHA, TRUE_BETA, num_seqs=200, max_time=100.0, seed=42)
    test_raw = simulate_hawkes_clean(TRUE_MU, TRUE_ALPHA, TRUE_BETA, num_seqs=50, max_time=100.0, seed=123)
    
    avg_len = np.mean([len(s) for s in train_raw])
    print(f"\n📊 Avg sequence length: {avg_len:.1f}")
    
    batch_train = prepare_batch(train_raw)
    batch_test = prepare_batch(test_raw)
    
    # --- Setup Model ---
    config = Config()
    model = Hawkes(config)
    
    print("\n🔹 Initial Parameters (Random):")
    print(f"   Mu:    {model.mu.item():.4f}")
    print(f"   Alpha: {model.alpha.item():.4f}")
    print(f"   Beta:  {model.beta.item():.4f}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Standard LR
    
    # --- Training Loop ---
    print("\n🚀 Starting Training...")
    
    for epoch in range(1, 301):
        optimizer.zero_grad()
        
        # Forward pass
        loss, num_events = model.loglike_loss(batch_train)
        
        # Calc NLL per event
        nll = loss / num_events
        
        loss.backward()
        
        # Clip gradients to ensure stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if epoch % 50 == 0:
            with torch.no_grad():
                loss_test, num_test = model.loglike_loss(batch_test)
                nll_test = loss_test / num_test
                
            print(f"Ep {epoch:03d} | Train NLL: {nll.item():.4f} | Test NLL: {nll_test.item():.4f} | "
                  f"Params: μ={model.mu.item():.3f} α={model.alpha.item():.3f} β={model.beta.item():.3f}")

    # --- Final Check ---
    print("\n" + "="*50)
    print("🏆 FINAL RESULTS")
    print("="*50)
    print(f"True:    μ={TRUE_MU}, α={TRUE_ALPHA}, β={TRUE_BETA}")
    print(f"Learned: μ={model.mu.item():.4f}, α={model.alpha.item():.4f}, β={model.beta.item():.4f}")
    
    # Metric
    mae = (abs(model.mu.item() - TRUE_MU) + 
           abs(model.alpha.item() - TRUE_ALPHA) + 
           abs(model.beta.item() - TRUE_BETA)) / 3
           
    print(f"\n📉 MAE (Mean Absolute Error): {mae:.4f}")
    
    if mae < 0.1:
        print("✅ SUCCESS: Model learned the dynamics!")
    else:
        print("⚠️  PARTIAL: Converged but with bias (common in short sequences).")

if __name__ == "__main__":
    main()
