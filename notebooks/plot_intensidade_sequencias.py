"""
Plota a intensidade estimada de sequências representativas de cada dataset.

Para cada dataset:
  - Esquerda: eventos no tempo (barras) + intensidade suavizada (KDE)
  - Direita: inter-event times (Δt) ao longo da sequência

Execução:
    python3.11 notebooks/plot_intensidade_sequencias.py
"""

import json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Funções
# ─────────────────────────────────────────────────────────────────────────────

def kde_intensity(event_times, bandwidth=None, n_points=500):
    """Estima intensidade λ(t) via Gaussian KDE sobre os tempos de evento."""
    t = np.array(event_times)
    if len(t) < 3:
        return np.array([]), np.array([])
    t_min, t_max = t.min(), t.max()
    if bandwidth is None:
        bandwidth = (t_max - t_min) / (len(t) ** 0.5)  # regra de bolso
        bandwidth = max(bandwidth, 1e-6)
    grid = np.linspace(t_min, t_max, n_points)
    # KDE manual (Gaussiana)
    kde = np.zeros_like(grid)
    for ti in t:
        kde += np.exp(-0.5 * ((grid - ti) / bandwidth) ** 2)
    kde /= (bandwidth * np.sqrt(2 * np.pi))
    return grid, kde


def pick_representative_seq(seqs_times, target_len_percentile=50):
    """Escolhe uma sequência com comprimento próximo da mediana."""
    lens = np.array([len(s) for s in seqs_times])
    target = np.percentile(lens, target_len_percentile)
    idx = np.argmin(np.abs(lens - target))
    return seqs_times[idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# Carrega datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_times_hf(hf_name, split='train', max_seqs=2000):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=split)
    out = []
    for i, row in enumerate(ds):
        if i >= max_seqs:
            break
        t = row.get('time_since_start', [])
        if len(t) >= 10:
            out.append(np.array(t, dtype=np.float64))
    return out


def simulate_hawkes_n_events(mu, alpha, beta, n_events, n_seqs, rng):
    all_seqs = []
    for _ in range(n_seqs):
        events = [0.0]
        t = 0.0
        lam_star = mu
        while len(events) < n_events + 1:
            w = rng.exponential(1.0 / lam_star)
            t_cand = t + w
            hist = np.array(events)
            lam_t = mu + alpha * np.sum(np.exp(-beta * (t_cand - hist)))
            if rng.uniform() <= lam_t / lam_star:
                events.append(t_cand)
                lam_star = lam_t + alpha
                t = t_cand
            else:
                lam_star = lam_t
                t = t_cand
        all_seqs.append(np.array(events[1:]))
    return all_seqs


DATASETS_HF = {
    'amazon':        'easytpp/amazon',
    'taxi':          'easytpp/taxi',
    'stackoverflow': 'easytpp/stackoverflow',
    'retweet':       'easytpp/retweet',
}

OUTCOME = {
    'hawkes':        'WIN',
    'amazon':        'WIN',
    'taxi':          'TIE',
    'stackoverflow': 'LOSE',
    'retweet':       'LOSE',
}
OC_COLOR = {'WIN': '#2e7d32', 'TIE': '#f57f17', 'LOSE': '#c62828'}

all_times = {}

# Hawkes sintético
print(f"[{time.time()-t0:.0f}s] Simulando Hawkes β=0.50...", flush=True)
rng = np.random.default_rng(42)
all_times['hawkes'] = simulate_hawkes_n_events(
    mu=0.2, alpha=0.8, beta=0.50, n_events=80, n_seqs=100, rng=rng)

# Reais
try:
    from datasets import load_dataset
    for name, hf_name in DATASETS_HF.items():
        print(f"[{time.time()-t0:.0f}s] Carregando {name}...", flush=True)
        all_times[name] = load_times_hf(hf_name, max_seqs=500)
except ImportError:
    print("HuggingFace não disponível", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot principal
# ─────────────────────────────────────────────────────────────────────────────

DS_ORDER = [n for n in ['hawkes', 'amazon', 'taxi', 'stackoverflow', 'retweet']
            if n in all_times]

n_ds = len(DS_ORDER)
fig, axes = plt.subplots(n_ds, 3, figsize=(18, 3.5 * n_ds),
                         gridspec_kw={'width_ratios': [3, 1.5, 1.5]})

for row, name in enumerate(DS_ORDER):
    seqs = all_times[name]
    seq, seq_idx = pick_representative_seq(seqs, target_len_percentile=50)

    # Normaliza pela média dos gaps (para comparabilidade)
    gaps = np.diff(seq)
    mean_gap = gaps[gaps > 0].mean() if np.any(gaps > 0) else 1.0
    seq_norm = (seq - seq[0]) / mean_gap
    dts = np.diff(seq_norm)

    oc = OUTCOME.get(name, '?')
    color = OC_COLOR.get(oc, '#555')

    # ── Coluna 1: Timeline + intensidade ──
    ax = axes[row, 0]

    # Eventos como barras verticais
    ax.vlines(seq_norm, 0, 0.15, colors=color, alpha=0.6, linewidth=0.8)

    # Intensidade suavizada (KDE)
    bw = max(mean_gap * 0.5, (seq_norm[-1] - seq_norm[0]) / (len(seq_norm) ** 0.6))
    grid, intensity = kde_intensity(seq_norm, bandwidth=bw)
    if len(grid) > 0:
        # Escala para ficar visível
        intensity_scaled = intensity / intensity.max() * 0.85
        ax.fill_between(grid, 0, intensity_scaled, color=color, alpha=0.15)
        ax.plot(grid, intensity_scaled, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlim(seq_norm[0] - 1, seq_norm[-1] + 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Tempo normalizado (t / mean_gap)', fontsize=8)
    ax.set_ylabel('Intensidade (KDE)', fontsize=8)
    ax.set_title(f'{name}  [{oc}]  —  seq #{seq_idx}, {len(seq)} eventos',
                 color=color, fontweight='bold', fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

    # ── Coluna 2: Δt ao longo da sequência ──
    ax2 = axes[row, 1]
    ax2.bar(range(len(dts)), dts, color=color, alpha=0.7, width=1.0, edgecolor='none')
    ax2.set_xlabel('Índice do evento', fontsize=8)
    ax2.set_ylabel('Δt (normalizado)', fontsize=8)
    ax2.set_title(f'Inter-event times', fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.15, axis='y')
    # Marca a mediana
    med_dt = np.median(dts)
    ax2.axhline(med_dt, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
                label=f'median={med_dt:.2f}')
    ax2.legend(fontsize=7)

    # ── Coluna 3: Histograma dos Δt (com fit exponencial visual) ──
    ax3 = axes[row, 2]
    dts_pos = dts[dts > 0]
    if len(dts_pos) > 5:
        bins = min(30, len(dts_pos) // 3)
        counts, edges, _ = ax3.hist(dts_pos, bins=bins, density=True,
                                     color=color, alpha=0.6, edgecolor='white')
        # Referência exponencial (rate = 1/mean)
        x_ref = np.linspace(0, np.percentile(dts_pos, 98), 200)
        rate = 1.0 / dts_pos.mean()
        ax3.plot(x_ref, rate * np.exp(-rate * x_ref), 'k--', linewidth=1.5,
                 alpha=0.7, label=f'Exp(λ={rate:.2f})')
        ax3.set_xlabel('Δt', fontsize=8)
        ax3.set_ylabel('Densidade', fontsize=8)
        ax3.set_title(f'Distribuição dos Δt', fontsize=9)
        ax3.legend(fontsize=7)
        ax3.tick_params(labelsize=7)
        ax3.grid(True, alpha=0.15)

    # Stats no canto
    cv = dts_pos.std() / dts_pos.mean() if dts_pos.mean() > 0 else 0
    ax3.text(0.95, 0.95, f'CV={cv:.2f}\nn={len(dts_pos)}',
             transform=ax3.transAxes, fontsize=7, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle(
    'Intensidade e Distribuição Temporal — Sequências Representativas\n'
    'Esquerda: timeline + λ(t) estimada  |  Centro: Δt por evento  |  '
    'Direita: histograma Δt vs exponencial',
    fontsize=11, y=1.01
)
plt.tight_layout()
out_path = 'notebooks/fig_intensidade_sequencias.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigura salva: {out_path}", flush=True)
print(f"[Total: {time.time()-t0:.0f}s]", flush=True)
