"""Gera HoTHP_validation_real_dataset.ipynb conforme o spec.
Cada celula e uma raw-string (r'''...''') para preservar barras e chaves literais.
"""
import json, pathlib

cells = []

def md(src):  cells.append(('markdown', src))
def code(src): cells.append(('code', src))

# ─────────────────────────────────────────────────────────────────────────────
md(r'''# Validação HoTHP — Datasets Reais (EasyTPP)

Compara **NHP · THP · RoTHP · HoTHP** em datasets reais do EasyTPP sob o protocolo
de extrapolação **"train-short / test-long"**: treina com sequências truncadas em
`train_len` eventos e avalia em `fator × train_len` (fatores **1×, 2×, 5×**).

Análogo a `HoTHP_validation_synthetic_dataset.ipynb` — mesmas métricas (NLL, RMSE,
Accuracy), mesmas cores por modelo, figuras comparáveis.

**Características**
- Motor: Runner do EasyTPP (mesmo caminho que gerou as tabelas da dissertação).
- Persistência no **Google Drive**: checkpoints + `metrics_cache.json` + `results.json`
  + figuras. Cada `(dataset, variante, modelo, seed, fator)` é uma chave isolada no
  cache — rodar um dataset hoje e outro amanhã **só acrescenta**, nunca sobrescreve.
- Execução **cadenciada**: comente/descomente datasets; o que já foi treinado é pulado.
- **Retweet** roda em duas variantes: `raw` (cru, reproduz o colapso numérico) e
  `norm` (timestamps ÷ média dos gaps por sequência).

**Colab:** Runtime → Change runtime type → T4 GPU.''')

# ── CELL: setup ──────────────────────────────────────────────────────────────
code(r'''# ── Setup (clone do repo, deps, fixes de import) ─────────────────────────────
import os, sys, json, time, pathlib

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
REPO_DIR = 'ufc-easytpp'

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        os.system('git clone https://github.com/hugoramos/ufc-easytpp.git')
    else:
        os.system('git -C ufc-easytpp reset --hard -q && git -C ufc-easytpp pull -q')
    os.system('pip install omegaconf datasets pyyaml matplotlib pandas seaborn -q')

    # __init__.py minimo (inclui NHP) — evita imports de modelos com deps extras
    init_path = os.path.join(REPO_DIR, 'easy_tpp/model/__init__.py')
    with open(init_path, 'w') as f:
        f.write(
            "from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel\n"
            "from easy_tpp.model.torch_model.torch_nhp   import NHP    as TorchNHP\n"
            "from easy_tpp.model.torch_model.torch_thp   import THP    as TorchTHP\n"
            "from easy_tpp.model.torch_model.torch_rothp import RoTHP  as TorchRoTHP\n"
            "from easy_tpp.model.torch_model.torch_hothp import HoTHP  as TorchHoTHP\n"
        )
    # generate_model_from_config() usa __subclasses__(): forca o import de todos os modelos
    runner_path = os.path.join(REPO_DIR, 'easy_tpp/runner/tpp_runner.py')
    with open(runner_path) as f:
        rc = f.read()
    if 'import easy_tpp.model  # noqa' not in rc:
        rc = rc.replace('from collections import OrderedDict\n',
                        'from collections import OrderedDict\nimport easy_tpp.model  # noqa: F401\n', 1)
        with open(runner_path, 'w') as f:
            f.write(rc)
    sys.path.insert(0, os.path.abspath(REPO_DIR))
else:
    # roda local: sobe ate achar a raiz do repo (pasta que contem easy_tpp/)
    _p = pathlib.Path.cwd()
    for _ in range(6):
        if (_p / 'easy_tpp').is_dir():
            break
        _p = _p.parent
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
GPU = 0 if torch.cuda.is_available() else -1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('IN_COLAB =', IN_COLAB, '| GPU =', GPU, '| torch =', torch.__version__)''')

# ── CELL: verifica patches + monta Drive ─────────────────────────────────────
code(r'''# ── Verifica patches do modelo e monta o Drive ───────────────────────────────
# O repo ja deve estar no estado corrigido (igual ao notebook sintetico final):
#   - attention usa masked_fill(..., -1e4)   (estabilidade numerica)
#   - HoTHP._normalize_timestamps e no-op: return time_seqs - time_seqs[:, :1]
import inspect
import easy_tpp.model.torch_model.torch_baselayer as _bl
import easy_tpp.model.torch_model.torch_hothp as _hh

_attn_src = inspect.getsource(_bl.attention)
_norm_src = inspect.getsource(_hh.HoTHP._normalize_timestamps)
assert '-1e4' in _attn_src, 'patch de attention (-1e4) ausente no repo'
assert 'time_seqs - time_seqs[:, :1]' in _norm_src, 'patch _normalize_timestamps (no-op) ausente'
print('OK patches: attention(-1e4) e _normalize_timestamps(no-op) presentes')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = pathlib.Path('/content/drive/MyDrive/HoTHP_validation_real_dataset')
else:
    BASE_DIR = pathlib.Path('.').resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR       = BASE_DIR / 'checkpoints'
NORM_DIR       = BASE_DIR / 'datasets_norm'
FIG_DIR        = BASE_DIR / 'figures'
for d in (CKPT_DIR, NORM_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
CACHE_PATH     = BASE_DIR / 'metrics_cache.json'
RESULTS_PATH   = BASE_DIR / 'results.json'
TRAINTIME_PATH = BASE_DIR / 'train_times.json'
print('BASE_DIR =', BASE_DIR)''')

# ── CELL: imports + config global ────────────────────────────────────────────
code(r'''# ── Imports e configuracao global ────────────────────────────────────────────
import gc, tempfile, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# Hiperparametros (metodologia da dissertacao, dados reais)
SEEDS          = [42, 142, 242]   # mesma formula do notebook sintetico (42 + i*100), 3 primeiras
MODELS         = ['NHP', 'THP', 'RoTHP', 'HoTHP']   # modelos treinados/avaliados
# Modelos OCULTOS nas tabelas/figuras (continuam no cache; nao re-treina nada).
# Deixe vazio ([]) para mostrar todos. NHP e recorrente: sua NLL nao e diretamente
# comparavel aos transformers (ele ganha na densidade temporal e perde em accuracy).
EXCLUDE_FROM_PLOTS = ['NHP']
PLOT_MODELS    = [m for m in MODELS if m not in EXCLUDE_FROM_PLOTS]
EXTRAP_FACTORS = [1, 2, 5]
MAX_EPOCH      = 100
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 256
HIDDEN_SIZE    = 64
NUM_HEADS      = 2
NUM_LAYERS     = 2
DROPOUT        = 0.1
TIME_EMB_SIZE  = 16
USE_LN         = False

CORES = {'NHP': '#55A868', 'THP': '#8172B2', 'RoTHP': '#4C72B0', 'HoTHP': '#C44E52'}

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('modelos:', MODELS, '| seeds:', SEEDS, '| fatores:', EXTRAP_FACTORS)''')

# ── CELL: DATASETS toggle ────────────────────────────────────────────────────
code(r'''# ── Datasets (comente/descomente para ativar/desativar) ──────────────────────
# train_len e num_event_types seguem a metodologia da dissertacao.
# variants: 'raw' usa o dataset do HuggingFace cru; 'norm' usa versao local
#           normalizada (todos os timestamps de cada seq divididos pela media dos gaps).
# max_real: maior comprimento de sequencia observado (limita os fatores viaveis).

DATASETS = {
    'amazon':        dict(num_event_types=16, pad_token_id=16, train_len=18,
                          max_real=94,  variants=['raw']),
    'stackoverflow': dict(num_event_types=22, pad_token_id=22, train_len=20,
                          max_real=101, variants=['raw']),
    'taxi':          dict(num_event_types=10, pad_token_id=10, train_len=7,
                          max_real=38,  variants=['raw']),
    'retweet':       dict(num_event_types=3,  pad_token_id=3,  train_len=50,
                          max_real=264, variants=['raw', 'norm']),

    # ── desativados por padrao (descomente para incluir) ──
    # 'taobao':     dict(num_event_types=17, pad_token_id=17, train_len=20,
    #                    max_real=64,  variants=['raw']),   # confirme tipos/max_real na celula de analise
    # 'earthquake': dict(num_event_types=7,  pad_token_id=7,  train_len=10,
    #                    max_real=18,  variants=['raw']),   # seqs curtas: tende a so 1x
}

def feasible_factors(ds):
    """Fatores cujo comprimento de teste cabe no maior comprimento real do dataset."""
    return [f for f in EXTRAP_FACTORS if f * ds['train_len'] <= ds['max_real']]

print(f'{"dataset":<16}{"tipos":>6}{"train_len":>11}{"variants":>16}   fatores')
print('-' * 70)
for name, ds in DATASETS.items():
    print(f'{name:<16}{ds["num_event_types"]:>6}{ds["train_len"]:>11}'
          f'{",".join(ds["variants"]):>16}   {feasible_factors(ds)}')''')

# ── CELL: analise de escala ──────────────────────────────────────────────────
code(r'''# ── Analise de escala temporal (confirme specs antes de treinar) ─────────────
print(f'{"dataset":<16}{"n_seqs":>8}{"len_min":>8}{"len_med":>8}{"len_max":>8}'
      f'{"mean_dt":>12}{"max_dt":>14}{"dim":>6}')
print('-' * 84)
for name in DATASETS:
    try:
        ds = load_dataset(f'easytpp/{name}', split='train', trust_remote_code=True)
        lens, all_dt, dim = [], [], None
        for row in ds:
            lens.append(len(row.get('time_since_start', [])))
            for d in row.get('time_since_last_event', []):
                if d > 0:
                    all_dt.append(d)
            if dim is None:
                dim = row.get('dim_process', None)
        lens = np.array(lens); all_dt = np.array(all_dt, dtype=np.float64)
        print(f'{name:<16}{len(lens):>8}{lens.min():>8}{int(np.median(lens)):>8}{lens.max():>8}'
              f'{all_dt.mean():>12.4f}{all_dt.max():>14.4f}{str(dim):>6}')
    except Exception as e:
        print(f'{name:<16}  ERRO: {e}')
print('\nConfira: num_event_types == dim, e train_len*max(fator) <= len_max para extrapolar.')''')

# ── CELL: prepara variantes norm ─────────────────────────────────────────────
code(r'''# ── Prepara variantes 'norm' (timestamps / media dos gaps por sequencia) ─────
# Salva JSON local no Drive (train/validation/test). So gera o que faltar.

def prepare_norm(name):
    out = NORM_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    files = {'train': out / 'train.json', 'dev': out / 'validation.json', 'test': out / 'test.json'}
    if all(p.exists() for p in files.values()):
        return files
    split_map = {'train': 'train', 'dev': 'validation', 'test': 'test'}
    for split, fpath in files.items():
        if fpath.exists():
            continue
        try:
            raw = load_dataset(f'easytpp/{name}', split=split_map[split], trust_remote_code=True)
        except Exception:
            raw = load_dataset(f'easytpp/{name}', split='train', trust_remote_code=True)
        seqs = []
        for row in raw:
            t = np.array(row['time_since_start'], dtype=np.float64)
            types = list(row['type_event'])
            dim = row.get('dim_process', (max(types) + 1 if types else 1))
            if len(t) < 3:
                continue
            gaps = np.diff(t)
            mg = gaps[gaps > 0].mean() if np.any(gaps > 0) else 1.0
            if mg <= 0:
                mg = 1.0
            tn = (t - t[0]) / mg
            dtn = np.insert(np.diff(tn), 0, 0.0)
            seqs.append({'time_since_start': tn.tolist(),
                         'time_since_last_event': dtn.tolist(),
                         'type_event': types, 'dim_process': int(dim)})
        with open(fpath, 'w') as f:
            json.dump(seqs, f)
        print(f'  {name}/{split}: {len(seqs)} seqs -> {fpath.name}')
    return files

for name, ds in DATASETS.items():
    if 'norm' in ds['variants']:
        print(f'[norm] {name} ...')
        prepare_norm(name)
print('OK variantes norm prontas')''')

# ── CELL: helpers ────────────────────────────────────────────────────────────
code(r'''# ── Helpers: config YAML, descoberta de checkpoint, cache ────────────────────

def variant_dirs(name, variant):
    if variant == 'raw':
        src = f'easytpp/{name}'
        return src, src, src
    p = NORM_DIR / name
    return str(p / 'train.json'), str(p / 'validation.json'), str(p / 'test.json')

def make_yaml(name, variant, model_id, seed, max_len, stage='train', pretrained_model_dir=None):
    ds = DATASETS[name]
    exp_id = f'{model_id}_{stage}'
    tr, va, te = variant_dirs(name, variant)
    model_cfg = {
        'hidden_size': HIDDEN_SIZE, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS,
        'dropout': DROPOUT, 'time_emb_size': TIME_EMB_SIZE, 'use_ln': USE_LN,
        'thinning': {'num_sample': 1, 'num_exp': 500, 'look_ahead_time': 10,
                     'patience_counter': 5, 'over_sample_rate': 5, 'num_samples_boundary': 5,
                     'dtime_max': 10, 'num_seq': 10, 'num_step_gen': 1},
    }
    if stage == 'train':
        model_cfg['loss_integral_num_sample_per_step'] = 20
        model_cfg['mc_num_sample_per_step'] = 20
    if pretrained_model_dir:
        model_cfg['pretrained_model_dir'] = pretrained_model_dir

    trainer_cfg = {'batch_size': BATCH_SIZE, 'max_epoch': MAX_EPOCH if stage == 'train' else 1,
                   'seed': seed, 'gpu': GPU, 'metrics': ['acc', 'rmse']}
    if stage == 'train':
        trainer_cfg.update({'valid_freq': 1, 'use_tfb': False, 'optimizer': 'adam',
                            'learning_rate': LEARNING_RATE, 'shuffle': False})

    base_dir = str(CKPT_DIR / f'{name}__{variant}' / model_id / f'seed{seed}') + '/'
    config = {
        'pipeline_config_id': 'runner_config',
        'data': {name: {'data_format': 'json', 'train_dir': tr, 'valid_dir': va, 'test_dir': te,
                        'data_specs': {'num_event_types': ds['num_event_types'],
                                       'pad_token_id': ds['pad_token_id'],
                                       'padding_side': 'right', 'truncation_side': 'right',
                                       'truncation_strategy': 'longest_first', 'max_len': max_len}}},
        exp_id: {'base_config': {'stage': stage, 'backend': 'torch', 'dataset_id': name,
                                 'runner_id': 'std_tpp', 'model_id': model_id, 'base_dir': base_dir},
                 'trainer_config': trainer_cfg, 'model_config': model_cfg},
    }
    return config, exp_id

def load_cfg(config_dict, exp_id):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        tmp = f.name
    try:
        return Config.build_from_yaml_file(tmp, experiment_id=exp_id)
    finally:
        os.unlink(tmp)

def get_model_dir(name, variant, model_id, seed):
    """EasyTPP salva em {base_dir}/{unique_id}/models/saved_model. Escaneia o mais recente."""
    base = CKPT_DIR / f'{name}__{variant}' / model_id / f'seed{seed}'
    if not base.is_dir():
        return str(base / 'models' / 'saved_model')
    cands = []
    for entry in os.scandir(base):
        if entry.is_dir():
            c = os.path.join(entry.path, 'models', 'saved_model')
            if os.path.exists(c):
                cands.append((entry.stat().st_mtime, c))
    return sorted(cands)[-1][1] if cands else str(base / 'models' / 'saved_model')

# ── Cache de metricas: fonte da verdade, escrita por chave (nunca sobrescreve) ─
def load_json(path, default):
    if pathlib.Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

metrics_cache = load_json(CACHE_PATH, {})
train_times   = load_json(TRAINTIME_PATH, {})

def dv_key(name, variant):
    return f'{name}__{variant}'

def cache_get(name, variant, model, seed, factor):
    return metrics_cache.get(dv_key(name, variant), {}).get(model, {}) \
        .get(str(seed), {}).get(str(factor))

def cache_put(name, variant, model, seed, factor, vals):
    metrics_cache.setdefault(dv_key(name, variant), {}).setdefault(model, {}) \
        .setdefault(str(seed), {})[str(factor)] = vals
    save_json(CACHE_PATH, metrics_cache)

combos = [(n, v) for n, ds in DATASETS.items() for v in ds['variants']]
print('OK helpers | combinacoes dataset/variant:', combos)''')

# ── CELL: treino ─────────────────────────────────────────────────────────────
code(r'''# ── Treino: (dataset, variante, modelo, seed) com max_len = train_len ────────
# Pula o que ja tem checkpoint fisico no disco (resiliente a reset do Colab).

done = skipped = 0
for name, variant in combos:
    ds = DATASETS[name]
    for model_id in MODELS:
        for seed in SEEDS:
            mdir = get_model_dir(name, variant, model_id, seed)
            if os.path.exists(mdir):
                skipped += 1
                print(f'[SKIP] {name}/{variant}/{model_id}/seed{seed}')
                continue
            print(f'\n=== treino {name}/{variant}/{model_id}/seed{seed} '
                  f'(max_len={ds["train_len"]}) ===')
            cfg_d, exp = make_yaml(name, variant, model_id, seed, ds['train_len'], stage='train')
            runner = Runner.build_from_config(load_cfg(cfg_d, exp))
            t0 = time.time()
            runner.run()
            dt = time.time() - t0
            train_times[f'{dv_key(name, variant)}|{model_id}|{seed}'] = dt
            save_json(TRAINTIME_PATH, train_times)
            del runner; free_gpu()
            done += 1
            print(f'[OK] {dt:.0f}s ({dt/60:.1f}min) -> {get_model_dir(name, variant, model_id, seed)}')
print(f'\nTreino concluido: {done} treinados, {skipped} pulados.')''')

# ── CELL: avaliacao ──────────────────────────────────────────────────────────
code(r'''# ── Avaliacao: cada modelo treinado em cada fator viavel ─────────────────────
# Grava no metrics_cache.json apos cada item; pula o que ja esta em cache.

for name, variant in combos:
    ds = DATASETS[name]
    factors = feasible_factors(ds)
    for model_id in MODELS:
        for seed in SEEDS:
            mdir = get_model_dir(name, variant, model_id, seed)
            if not os.path.exists(mdir):
                print(f'[WARN] sem checkpoint: {name}/{variant}/{model_id}/seed{seed}')
                continue
            for f in factors:
                if cache_get(name, variant, model_id, seed, f) is not None:
                    print(f'[SKIP] {name}/{variant}/{model_id}/seed{seed}/{f}x (cache)')
                    continue
                max_len = f * ds['train_len']
                cfg_d, exp = make_yaml(name, variant, model_id, seed, max_len,
                                       stage='eval', pretrained_model_dir=mdir)
                runner = Runner.build_from_config(load_cfg(cfg_d, exp))
                test_loader = runner._data_loader.test_loader()
                m = runner._evaluate_model(test_loader)
                del runner; free_gpu()
                vals = {'nll':  float(-m.get('loglike', float('nan'))),
                        'rmse': float(m.get('rmse', float('nan'))),
                        'acc':  float(m.get('acc', float('nan')))}
                cache_put(name, variant, model_id, seed, f, vals)
                print(f'{name}/{variant}/{model_id}/seed{seed}/{f}x -> '
                      f'NLL={vals["nll"]:.4f}  RMSE={vals["rmse"]:.4f}  ACC={vals["acc"]:.4f}')
print('\nOK avaliacao concluida')''')

# ── CELL: reconstroi all_results + results.json ──────────────────────────────
code(r'''# ── all_results a partir do metrics_cache COMPLETO ───────────────────────────
# Le TUDO que ja foi computado em qualquer sessao, independente de quais datasets
# estao ativos agora. Assim voce pode comentar/descomentar datasets e ir rodando:
# as tabelas/figuras/results.json sempre refletem o conjunto completo acumulado.
METRIC_NAMES = ['nll', 'rmse', 'acc']

def build_all_results():
    res = {}
    for key, models in metrics_cache.items():
        factors = set()
        for mdl, seeds in models.items():
            for sd, facs in seeds.items():
                factors.update(int(f) for f in facs)
        factors = sorted(factors)
        # ordena modelos conhecidos primeiro (MODELS), depois quaisquer extras do cache
        present = [m for m in MODELS if m in models] + [m for m in models if m not in MODELS]
        rd = {f: {mn: {met: [] for met in METRIC_NAMES} for mn in present} for f in factors}
        for mn in present:
            for sd, facs in models[mn].items():
                for f, vals in facs.items():
                    fi = int(f)
                    for met in METRIC_NAMES:
                        if met in vals:
                            rd[fi][mn][met].append(vals[met])
        res[key] = rd
    return res

all_results = build_all_results()

def panels():
    """Paineis = todos os dataset__variant presentes no cache."""
    return list(all_results.keys())

def models_in(rd):
    """Modelos a exibir num painel: presentes no cache, na ordem de MODELS,
    filtrados por PLOT_MODELS (oculta os de EXCLUDE_FROM_PLOTS, ex.: NHP)."""
    seen = set()
    for f in rd:
        seen.update(rd[f].keys())
    ordered = [m for m in MODELS if m in seen] + [m for m in seen if m not in MODELS]
    return [m for m in ordered if m in PLOT_MODELS]

save_json(RESULTS_PATH, {k: {str(f): v for f, v in rd.items()} for k, rd in all_results.items()})
print('datasets/variantes acumulados no results.json:', panels())
print('OK results.json salvo em', RESULTS_PATH)''')

# ── CELL: tabelas resumo ─────────────────────────────────────────────────────
code(r'''# ── Tabelas-resumo (media +/- desvio, por dataset/variante/fator) ────────────
def fmt(vals):
    a = np.array(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return '       nan'
    return f'{a.mean():.4f}+/-{a.std():.4f}'

for met, label in [('nll', 'NLL (menor=melhor)'),
                   ('rmse', 'RMSE (menor=melhor)'),
                   ('acc', 'Accuracy (maior=melhor)')]:
    print('\n' + '=' * 72)
    print(f'RESUMO — {label} (n={len(SEEDS)} seeds)')
    print('=' * 72)
    for key, rd in all_results.items():
        factors = sorted(rd.keys())
        print(f'\n{key}')
        print(f'  {"modelo":<8}' + ''.join(f'{str(f) + "x":>18}' for f in factors))
        for mn in models_in(rd):
            row = f'  {mn:<8}'
            for f in factors:
                row += f'{fmt(rd[f][mn][met]):>18}'
            print(row)''')

# ── CELL: figuras por metrica ────────────────────────────────────────────────
code(r'''# ── Figuras: metrica vs fator de extrapolacao (linhas com banda +/- std) ─────
def plot_metric(met, ylabel, marker, fname):
    P = panels()
    if not P:
        print('sem paineis'); return
    ncol = min(len(P), 3)
    nrow = int(np.ceil(len(P) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)
    for idx, key in enumerate(P):
        ax = axes[idx // ncol][idx % ncol]
        rd = all_results[key]
        factors = sorted(rd.keys())
        x = np.arange(len(factors))
        big = False
        for mn in models_in(rd):
            means = np.array([np.nanmean(rd[f][mn][met]) if len(rd[f][mn][met]) else np.nan
                              for f in factors])
            stds = np.array([np.nanstd(rd[f][mn][met]) if len(rd[f][mn][met]) else np.nan
                             for f in factors])
            if np.all(np.isnan(means)):
                continue
            ax.plot(x, means, marker, color=CORES.get(mn, '#777777'), lw=2, ms=7, label=mn)
            ax.fill_between(x, means - stds, means + stds, color=CORES.get(mn, '#777777'), alpha=0.15)
            if np.nanmax(np.abs(means)) > 100:
                big = True
        if big and met != 'acc':
            ax.set_yscale('symlog')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}x' for f in factors])
        ax.set_title(key, fontweight='bold', fontsize=10)
        ax.set_xlabel('fator de extrapolacao')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(len(P), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'{ylabel} por fator de extrapolacao ({len(SEEDS)} seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print('salvo:', FIG_DIR / fname)

plot_metric('nll',  'NLL (nats)', 'o-', 'real_nll.png')
plot_metric('rmse', 'RMSE',       's-', 'real_rmse.png')
plot_metric('acc',  'Accuracy',   '^-', 'real_acc.png')''')

# ── CELL: grid consolidado ───────────────────────────────────────────────────
code(r'''# ── Grid consolidado: 3 metricas x paineis ───────────────────────────────────
P = panels()
ncol = max(len(P), 1)
fig, axes = plt.subplots(3, ncol, figsize=(5 * ncol, 12), squeeze=False)
info = [('nll', 'NLL (nats)', 'o-'), ('rmse', 'RMSE', 's-'), ('acc', 'Accuracy', '^-')]
for r, (met, ylabel, marker) in enumerate(info):
    for c, key in enumerate(P):
        ax = axes[r][c]
        rd = all_results[key]
        factors = sorted(rd.keys())
        x = np.arange(len(factors))
        big = False
        for mn in models_in(rd):
            means = np.array([np.nanmean(rd[f][mn][met]) if len(rd[f][mn][met]) else np.nan
                              for f in factors])
            stds = np.array([np.nanstd(rd[f][mn][met]) if len(rd[f][mn][met]) else np.nan
                             for f in factors])
            if np.all(np.isnan(means)):
                continue
            ax.plot(x, means, marker, color=CORES.get(mn, '#777777'), lw=2, ms=6, label=mn)
            ax.fill_between(x, means - stds, means + stds, color=CORES.get(mn, '#777777'), alpha=0.15)
            if np.nanmax(np.abs(means)) > 100:
                big = True
        if big and met != 'acc':
            ax.set_yscale('symlog')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}x' for f in factors])
        if r == 0:
            ax.set_title(key, fontweight='bold', fontsize=10)
        if c == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
fig.suptitle(f'Consolidado — 3 metricas x paineis ({len(SEEDS)} seeds)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'real_consolidado.png', dpi=150, bbox_inches='tight')
plt.show()
print('salvo:', FIG_DIR / 'real_consolidado.png')''')

# ── CELL: tempos de treino ───────────────────────────────────────────────────
code(r'''# ── Tempos de treino ─────────────────────────────────────────────────────────
if train_times:
    rows = []
    for k, t in train_times.items():
        dvk, model_id, seed = k.split('|')
        rows.append({'dataset_variant': dvk, 'model': model_id, 'seed': int(seed),
                     's': t, 'min': t / 60})
    dft = pd.DataFrame(rows)
    g = dft.groupby(['dataset_variant', 'model'])['min'].agg(['mean', 'std', 'count']).reset_index()
    print(g.to_string(index=False))
    print(f'\nTotal acumulado: {dft["s"].sum() / 3600:.2f} h')
else:
    print('Sem tempos de treino registrados ainda.')''')

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    'cells': [
        {'cell_type': t,
         'metadata': {},
         'source': s.splitlines(keepends=True)} | (
            {'outputs': [], 'execution_count': None} if t == 'code' else {})
        for t, s in cells
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'colab': {'provenance': []},
        'accelerator': 'GPU',
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out = pathlib.Path(__file__).parent / 'HoTHP_validation_real_dataset.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('escrito:', out, '|', len(cells), 'celulas')
