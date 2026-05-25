"""
Células para colar no Colab (após o Benchmark_Real_Datasets_Colab).
Extrai e plota λ*(t) que cada modelo treinado calcula para sequências reais.

Mostra visualmente: o RoTHP aprende intensidades "não-Hawkes" para datasets
onde o processo não é Hawkes clássico.
"""

# ═══════════════════════════════════════════════════════════════════════
# CÉLULA 1: Funções de extração de intensidade
# ═══════════════════════════════════════════════════════════════════════

import torch
import numpy as np
import matplotlib.pyplot as plt
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def load_model_and_data(config_dict, experiment_id, pretrained_model_dir):
    """Carrega modelo treinado e retorna (model, test_loader)."""
    import copy
    cfg_dict = copy.deepcopy(config_dict)
    # Aponta para checkpoint
    for key in cfg_dict:
        if key.endswith('_eval') or key.endswith('_train') or '_' in key:
            if isinstance(cfg_dict[key], dict) and 'model_config' in cfg_dict[key]:
                cfg_dict[key]['model_config']['pretrained_model_dir'] = pretrained_model_dir
                cfg_dict[key]['base_config']['stage'] = 'eval'
                break

    import yaml, tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
        tmp_path = f.name
    cfg = Config.build_from_yaml_file(tmp_path, experiment_id=experiment_id)
    os.unlink(tmp_path)

    runner = Runner.build_from_config(cfg)
    model = runner._model._model  # TorchBaseModel instance
    model.eval()
    test_loader = runner._data_loader.test_loader()
    return model, test_loader, runner


def extract_intensity_curve(model, batch, n_samples_per_step=50):
    """
    Extrai λ*(t) em pontos densos entre cada par de eventos consecutivos.

    Args:
        model: TorchBaseModel treinado
        batch: um batch do data loader
        n_samples_per_step: quantos pontos amostrar entre cada par de eventos

    Returns:
        dict com:
          't_events': tempos dos eventos [seq_len]
          'types':    tipos dos eventos [seq_len]
          't_grid':   tempos dos pontos amostrados [n_points]
          'lambda_total': intensidade total λ*(t) = Σ_k λ_k*(t) [n_points]
          'lambda_per_type': intensidade por tipo [n_points, n_types]
    """
    time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch

    # Usa apenas a primeira sequência do batch
    B = 1
    time_seqs = time_seqs[:B].to(model.device)
    time_delta_seqs = time_delta_seqs[:B].to(model.device)
    type_seqs = type_seqs[:B].to(model.device)
    attention_mask = attention_mask[:B].to(model.device)
    batch_non_pad_mask = batch_non_pad_mask[:B].to(model.device)

    seq_len = int(batch_non_pad_mask[0].sum().item())
    t_events = time_seqs[0, :seq_len].cpu().numpy()
    types = type_seqs[0, :seq_len].cpu().numpy()
    dt = time_delta_seqs[0, :seq_len].cpu().numpy()

    # Para cada posição i, amostramos n pontos em [0, dt_{i+1}]
    # O modelo calcula λ*(t_i + τ) para τ ∈ sample_dtimes
    all_t = []
    all_lambda = []

    with torch.no_grad():
        # Forward pass para obter estados ocultos
        enc_out = model.forward(
            time_seqs[:, :seq_len],
            type_seqs[:, :seq_len],
            attention_mask[:, :seq_len, :seq_len]
        )

        for i in range(seq_len - 1):
            # Intervalo entre evento i e i+1
            dt_i = dt[i + 1]
            if dt_i <= 0:
                dt_i = 1e-4

            # Grid de pontos entre t_i e t_{i+1}
            taus = torch.linspace(0, dt_i, n_samples_per_step, device=model.device)

            # Computa intensidade: usa o estado do evento i
            # event_states: [1, 1, hidden_size]
            event_state_i = enc_out[:, i:i+1, :]  # [1, 1, hidden]

            # sample_dtimes precisa ser [1, 1, n_samples]
            sample_dt = taus.unsqueeze(0).unsqueeze(0)  # [1, 1, n_samples]

            # compute_states_at_sample_times espera:
            #   event_states: [B, seq_len, hidden] → usamos [1, 1, hidden]
            #   sample_dtimes: [B, seq_len, n_samples] → usamos [1, 1, n_samples]
            intensity_states = model.compute_states_at_sample_times(
                event_state_i, sample_dt
            )
            # intensity_states: [1, 1, n_samples, num_event_types]

            lambdas = model.softplus(intensity_states)
            # lambdas: [1, 1, n_samples, num_event_types]

            lambdas_np = lambdas[0, 0].cpu().numpy()  # [n_samples, num_types]
            t_points = t_events[i] + taus.cpu().numpy()

            all_t.append(t_points)
            all_lambda.append(lambdas_np)

    t_grid = np.concatenate(all_t)
    lambda_per_type = np.concatenate(all_lambda, axis=0)  # [n_points, n_types]
    lambda_total = lambda_per_type.sum(axis=1)  # [n_points]

    return {
        't_events': t_events,
        'types': types,
        't_grid': t_grid,
        'lambda_total': lambda_total,
        'lambda_per_type': lambda_per_type,
    }


print("Funções de extração definidas.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# CÉLULA 2: Extrair e plotar intensidades
# ═══════════════════════════════════════════════════════════════════════

# ── CONFIGURAÇÃO: ajustar caminhos conforme seu benchmark ──

# Datasets para plotar (use os que já treinaram no benchmark)
DATASETS_TO_PLOT = ['amazon', 'stackoverflow', 'retweet']

# Modelos para comparar
MODELS_TO_PLOT = ['RoTHP', 'HoTHP']

# Seed a usar (escolha um dos 3)
SEED = 2019

# Função auxiliar para encontrar checkpoint
# (ajustar conforme a estrutura do seu benchmark)
def find_checkpoint(dataset_name, model_id, seed):
    """Retorna o caminho do checkpoint treinado."""
    import glob
    base = f'./checkpoints/{dataset_name}/{model_id}/seed{seed}'
    # Procura pelo saved_model
    pattern = f'{base}/**/saved_model'
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return sorted(matches)[-1]  # mais recente
    # Tenta direto
    direct = f'{base}/models/saved_model'
    if os.path.exists(direct):
        return direct
    print(f"  WARN: checkpoint não encontrado em {base}", flush=True)
    return None


# ── Extrai intensidades ──
results = {}

for dataset_name in DATASETS_TO_PLOT:
    for model_id in MODELS_TO_PLOT:
        key = f"{dataset_name}_{model_id}"
        ckpt = find_checkpoint(dataset_name, model_id, SEED)
        if ckpt is None:
            continue

        print(f"Extraindo λ*(t) para {key}...", end=" ", flush=True)

        try:
            # Cria config de eval para este dataset/modelo
            # (reusa make_yaml do benchmark se disponível, ou cria manualmente)
            eval_max_len = DATASETS[dataset_name]['eval_5x']  # usa o max_len do 5x
            cfg_dict, exp_id = make_yaml(
                model_id=model_id,
                seed=SEED,
                dataset_name=dataset_name,
                max_len=eval_max_len,
                stage='eval',
            )
            cfg_dict[exp_id]['model_config']['pretrained_model_dir'] = ckpt

            model, test_loader, runner = load_model_and_data(
                cfg_dict, exp_id, ckpt)

            # Pega primeiro batch do test
            batch = next(iter(test_loader))

            # Extrai intensidade
            intensity_data = extract_intensity_curve(model, batch, n_samples_per_step=30)
            results[key] = intensity_data
            print(f"OK ({len(intensity_data['t_grid'])} pontos)", flush=True)

            del model, runner
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERRO: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# CÉLULA 3: Plot das intensidades
# ═══════════════════════════════════════════════════════════════════════

COLORS = {'RoTHP': '#2196F3', 'HoTHP': '#4CAF50', 'THP': '#FF9800'}

n_datasets = len(DATASETS_TO_PLOT)
fig, axes = plt.subplots(n_datasets, 1, figsize=(14, 4 * n_datasets))
if n_datasets == 1:
    axes = [axes]

for row, dataset_name in enumerate(DATASETS_TO_PLOT):
    ax = axes[row]

    for model_id in MODELS_TO_PLOT:
        key = f"{dataset_name}_{model_id}"
        if key not in results:
            continue

        data = results[key]
        color = COLORS.get(model_id, '#555')

        # Plota λ*(t) total
        ax.plot(data['t_grid'], data['lambda_total'],
                color=color, linewidth=1.5, alpha=0.8, label=model_id)

    # Marca eventos como barras verticais (usando primeiro modelo disponível)
    any_key = f"{dataset_name}_{MODELS_TO_PLOT[0]}"
    if any_key in results:
        t_ev = results[any_key]['t_events']
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.vlines(t_ev, 0, ymax * 0.05, colors='black', alpha=0.3,
                  linewidth=0.5, label='eventos')

    ax.set_xlabel('Tempo', fontsize=10)
    ax.set_ylabel('λ*(t) total', fontsize=10)
    ax.set_title(f'{dataset_name} — Intensidade aprendida por cada modelo (seed={SEED})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

fig.suptitle(
    'Intensidade Condicional λ*(t) — Modelos Treinados\n'
    'Comparação visual: como cada modelo "vê" a dinâmica temporal',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('intensidade_modelos_comparacao.png', dpi=150, bbox_inches='tight')
plt.show()
print("Salvo: intensidade_modelos_comparacao.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# CÉLULA 4 (BONUS): Zoom nos bursts — retweet
# ═══════════════════════════════════════════════════════════════════════

if 'retweet_RoTHP' in results and 'retweet_HoTHP' in results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, model_id in zip(axes, ['RoTHP', 'HoTHP']):
        data = results[f'retweet_{model_id}']
        color = COLORS[model_id]

        ax.plot(data['t_grid'], data['lambda_total'],
                color=color, linewidth=1.5)
        ax.vlines(data['t_events'], 0, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.1,
                  colors='black', alpha=0.3, linewidth=0.5)

        # Zoom no primeiro burst (primeiros 20% da sequência)
        t_max = data['t_events'][min(10, len(data['t_events'])-1)]
        ax.set_xlim(0, t_max * 1.1)

        ax.set_title(f'Retweet — {model_id} (zoom no burst inicial)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('λ*(t)')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('intensidade_retweet_zoom.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Salvo: intensidade_retweet_zoom.png", flush=True)
