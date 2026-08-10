"""
Plota λ*(t) que cada modelo treinado calcula para sequências de teste.
Usa checkpoints do benchmark.

Execução:
    python3.11 notebooks/plot_model_intensity.py
"""

import os, sys, glob, json, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
import easy_tpp.model  # registra modelos

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_CFG = {
    'amazon': {
        'hf': 'easytpp/amazon', 'num_types': 16, 'max_len': 50,
    },
    'stackoverflow': {
        'hf': 'easytpp/stackoverflow', 'num_types': 22, 'max_len': 100,
    },
    'retweet': {
        'hf': 'easytpp/retweet', 'num_types': 3, 'max_len': 250,
    },
}

MODELS = ['THP', 'RoTHP', 'HoTHP']
SEED = 2019
DEVICE = 'cpu'
N_SAMPLES = 30  # pontos entre cada par de eventos

COLORS = {'THP': '#FF9800', 'RoTHP': '#2196F3', 'HoTHP': '#4CAF50'}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint(dataset, model_id, seed):
    base = f'checkpoints/{dataset}/{model_id}/seed{seed}'
    matches = glob.glob(f'{base}/**/saved_model', recursive=True)
    if matches:
        # Pega o mais antigo (treino, não eval) — o que tem o modelo salvo
        for m in sorted(matches):
            if os.path.getsize(m) > 1000:
                return m
    return None


def load_model(dataset, model_id, seed):
    """Carrega modelo do checkpoint usando state_dict diretamente."""
    ckpt_path = find_checkpoint(dataset, model_id, seed)
    if ckpt_path is None:
        return None

    # Precisamos criar o modelo com a config correta
    num_types = DATASETS_CFG[dataset]['num_types']

    # Cria config mínima via omegaconf
    from omegaconf import OmegaConf
    model_config = OmegaConf.create({
        'hidden_size': 64,
        'num_heads': 2,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'time_emb_size': 16,
        'use_ln': False,
        'num_event_types': num_types,
        'num_event_types_pad': num_types + 1,
        'pad_token_id': num_types,
        'event_cls_num': num_types,
        'pretrained_model_dir': ckpt_path,
        'gpu': -1,
        'use_mc_samples': False,
        'loss_integral_num_sample_per_step': 20,
        'mc_num_sample_per_step': 20,
        'thinning': {
            'num_sample': 1, 'num_exp': 500, 'look_ahead_time': 10,
            'patience_counter': 5, 'over_sample_rate': 5,
            'num_samples_boundary': 5, 'dtime_max': 10,
            'num_seq': 10, 'num_step_gen': 1,
        },
    })

    # Instancia o modelo
    model_cls = None
    for subcls in TorchBaseModel.__subclasses__():
        if subcls.__name__ == model_id:
            model_cls = subcls
            break
        for subsub in subcls.__subclasses__():
            if subsub.__name__ == model_id:
                model_cls = subsub
                break

    if model_cls is None:
        print(f"  Classe {model_id} não encontrada", flush=True)
        return None

    model = model_cls(model_config)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


def load_test_sequence(hf_name, seq_idx=0, max_len=100):
    """Carrega uma sequência de teste do HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(hf_name, split='test')
    row = ds[seq_idx]
    t = np.array(row['time_since_start'], dtype=np.float64)
    types = np.array(row['type_event'], dtype=np.int64)

    # Trunca se necessário
    if len(t) > max_len:
        t = t[:max_len]
        types = types[:max_len]

    # Computa delta times
    dt = np.zeros_like(t)
    dt[1:] = np.diff(t)

    return t, dt, types


def compute_intensity_curve(model, t, dt, types, n_samples=30):
    """
    Calcula λ*(t) do modelo em pontos densos entre eventos.

    Retorna: (t_grid, lambda_total, lambda_per_type)
    """
    seq_len = len(t)

    # Prepara tensores [1, seq_len]
    time_seqs = torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    time_delta_seqs = torch.tensor(dt, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    type_seqs = torch.tensor(types, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # Attention mask (causal)
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1
    ).unsqueeze(0).bool()

    all_t = []
    all_lambda = []

    with torch.no_grad():
        # Forward pass
        enc_out = model.forward(time_seqs, type_seqs, attn_mask)

        for i in range(seq_len - 1):
            dt_i = dt[i + 1]
            if dt_i <= 1e-8:
                dt_i = 1e-4

            # Grid de pontos entre t_i e t_{i+1}
            taus = torch.linspace(0, dt_i, n_samples, device=DEVICE)

            # Estado oculto do evento i
            event_state_i = enc_out[:, i:i+1, :]  # [1, 1, hidden]

            # sample_dtimes: [1, 1, n_samples]
            sample_dt = taus.unsqueeze(0).unsqueeze(0)

            # Computa intensidade
            intensity_states = model.compute_states_at_sample_times(
                event_state_i, sample_dt
            )
            lambdas = model.softplus(intensity_states)  # [1, 1, n_samples, num_types]

            lambdas_np = lambdas[0, 0].cpu().numpy()  # [n_samples, num_types]
            t_points = t[i] + taus.cpu().numpy()

            all_t.append(t_points)
            all_lambda.append(lambdas_np)

    t_grid = np.concatenate(all_t)
    lambda_per_type = np.concatenate(all_lambda, axis=0)
    lambda_total = lambda_per_type.sum(axis=1)

    return t_grid, lambda_total, lambda_per_type


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Escolhe sequência com padrão interessante (não a primeira, que pode ser atípica)
SEQ_IDX = 5

results = {}

for ds_name, ds_cfg in DATASETS_CFG.items():
    print(f"\n[{time.time()-t0:.0f}s] === {ds_name} ===", flush=True)

    # Carrega sequência de teste
    print(f"  Carregando sequência #{SEQ_IDX}...", flush=True)
    t, dt, types = load_test_sequence(
        ds_cfg['hf'], seq_idx=SEQ_IDX, max_len=ds_cfg['max_len']
    )
    print(f"  → {len(t)} eventos, t=[{t[0]:.2f}, {t[-1]:.2f}]", flush=True)

    for model_id in MODELS:
        key = f"{ds_name}_{model_id}"
        print(f"  {model_id}...", end=" ", flush=True)

        model = load_model(ds_name, model_id, SEED)
        if model is None:
            print("SKIP (checkpoint não encontrado)", flush=True)
            continue

        try:
            t_grid, lam_total, lam_per_type = compute_intensity_curve(
                model, t, dt, types, n_samples=N_SAMPLES
            )
            results[key] = {
                't_events': t,
                'types': types,
                't_grid': t_grid,
                'lambda_total': lam_total,
                'lambda_per_type': lam_per_type,
            }
            print(f"OK (λ min={lam_total.min():.4f} max={lam_total.max():.4f})", flush=True)
        except Exception as e:
            print(f"ERRO: {e}", flush=True)

        del model

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

ds_names = list(DATASETS_CFG.keys())
n_ds = len(ds_names)

fig, axes = plt.subplots(n_ds, 1, figsize=(16, 4.5 * n_ds))
if n_ds == 1:
    axes = [axes]

for row, ds_name in enumerate(ds_names):
    ax = axes[row]

    max_lam = 0
    for model_id in MODELS:
        key = f"{ds_name}_{model_id}"
        if key not in results:
            continue
        data = results[key]
        color = COLORS[model_id]
        ax.plot(data['t_grid'], data['lambda_total'],
                color=color, linewidth=1.5, alpha=0.85, label=model_id)
        max_lam = max(max_lam, np.percentile(data['lambda_total'], 98))

    # Eventos como barras verticais
    any_key = [f"{ds_name}_{m}" for m in MODELS if f"{ds_name}_{m}" in results]
    if any_key:
        t_ev = results[any_key[0]]['t_events']
        ax.vlines(t_ev, 0, max_lam * 0.08, colors='black', alpha=0.4,
                  linewidth=0.6, label='events')

    ax.set_xlabel('Time (absolute)', fontsize=10)
    ax.set_ylabel('λ*(t) = Σₖ λₖ*(t)', fontsize=10)
    ax.set_title(f'{ds_name.capitalize()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0, top=max_lam * 1.15 if max_lam > 0 else 1)

# No overall title: the description is written in the LaTeX caption.
plt.tight_layout()
out = 'notebooks/fig_model_intensity.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigura salva: {out}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot normalizado (escala temporal normalizada para comparar formas)
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(n_ds, 1, figsize=(16, 4.5 * n_ds))
if n_ds == 1:
    axes2 = [axes2]

for row, ds_name in enumerate(ds_names):
    ax = axes2[row]

    for model_id in MODELS:
        key = f"{ds_name}_{model_id}"
        if key not in results:
            continue
        data = results[key]
        color = COLORS[model_id]

        # Normaliza tempo pela média dos gaps
        t_ev = data['t_events']
        mean_gap = np.diff(t_ev).mean() if len(t_ev) > 1 else 1.0
        t_norm = (data['t_grid'] - t_ev[0]) / mean_gap

        ax.plot(t_norm, data['lambda_total'],
                color=color, linewidth=1.5, alpha=0.85, label=model_id)

    # Eventos normalizados
    any_key = [f"{ds_name}_{m}" for m in MODELS if f"{ds_name}_{m}" in results]
    if any_key:
        t_ev = results[any_key[0]]['t_events']
        mean_gap = np.diff(t_ev).mean() if len(t_ev) > 1 else 1.0
        t_ev_norm = (t_ev - t_ev[0]) / mean_gap
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.vlines(t_ev_norm, 0, ymax * 0.08, colors='black', alpha=0.4, linewidth=0.6)

    ax.set_xlabel('Normalized time (t / mean gap)', fontsize=10)
    ax.set_ylabel('λ*(t)', fontsize=10)
    ax.set_title(f'{ds_name.capitalize()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

# No overall title: the description is written in the LaTeX caption.
plt.tight_layout()
out2 = 'notebooks/fig_model_intensity_normalized.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Figura salva: {out2}", flush=True)

print(f"\n[Total: {time.time()-t0:.0f}s]", flush=True)
