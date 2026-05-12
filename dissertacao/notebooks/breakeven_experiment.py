"""
Break-even experiment: wall-clock forward-pass time vs sequence length.

Trains THP, RoTHP, and HoTHP on synthetic Hawkes sequences of lengths
[64, 128, 256, 512] (no extrapolation) and records seconds per forward pass.

Results answer: for what sequence length does training HoTHP on short
sequences become cheaper than training RoTHP on the full long sequence?

Usage:
    python3.9 notebooks/breakeven_experiment.py
"""
import sys
import os
import time
import json

# Add project root to path so easy_tpp symlink resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from easy_tpp.model.torch_model.torch_thp import THP
from easy_tpp.model.torch_model.torch_rothp import RoTHP
from easy_tpp.model.torch_model.torch_hothp import HoTHP
from easy_tpp.config_factory.model_config import ModelConfig


def make_config(num_event_types=1):
    return ModelConfig(
        hidden_size=64,
        time_emb_size=16,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        use_ln=False,
        num_event_types=num_event_types,
        num_event_types_pad=num_event_types + 1,
        event_pad_index=num_event_types,
        gpu=-1,  # CPU
    )


def make_batch(seq_len, batch_size=4, seed=42):
    """Return (time_seqs, type_seqs, attention_mask) on CPU."""
    torch.manual_seed(seed)
    # Random inter-event gaps ~ Exp(1), generated purely in torch
    gaps = torch.distributions.Exponential(1.0).sample((batch_size, seq_len))
    times = gaps.cumsum(dim=1).float()

    time_seqs = times                                         # [B, L]
    type_seqs = torch.zeros(batch_size, seq_len, dtype=torch.long)  # [B, L]

    # Causal mask: mask[b, i, j]=True means j is masked for query i
    # Mask future positions (j > i)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]

    return time_seqs, type_seqs, attention_mask


def time_model(model_cls, seq_len, n_warmup=5, n_measure=20):
    """Return mean seconds per forward pass."""
    cfg = make_config()
    model = model_cls(cfg)
    model.eval()

    time_seqs, type_seqs, attention_mask = make_batch(seq_len)

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            model(time_seqs, type_seqs, attention_mask)

        # Measure
        t0 = time.perf_counter()
        for _ in range(n_measure):
            model(time_seqs, type_seqs, attention_mask)
        elapsed = time.perf_counter() - t0

    return elapsed / n_measure


def main():
    seq_lengths = [64, 128, 256, 512]
    models = [('THP', THP), ('RoTHP', RoTHP), ('HoTHP', HoTHP)]
    results = {}

    print(f"{'Model':8s}  {'L':>6s}  {'ms/fwd':>10s}")
    print('-' * 30)

    for name, cls in models:
        results[name] = {}
        for L in seq_lengths:
            t = time_model(cls, L)
            results[name][str(L)] = round(t * 1000, 2)  # store as ms
            print(f"{name:8s}  {L:6d}  {t*1000:10.1f} ms")

    print()
    print("HoTHP / RoTHP time ratio:")
    for L in seq_lengths:
        ratio = results['HoTHP'][str(L)] / results['RoTHP'][str(L)]
        print(f"  L={L}: {ratio:.2f}x")

    out_path = os.path.join(os.path.dirname(__file__), 'breakeven_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
