"""Gera HoTHP_vs_NHP_Intensity_Amazon.ipynb — a 'prova dos nove' da bimodalidade."""
import json, pathlib

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": s}

cells = []

cells.append(md(r"""# Prova dos nove: intensidade do NHP vs HoTHP no Amazon

Objetivo: **ver com os próprios olhos** a bimodalidade dos intervalos do Amazon sendo
tratada pelo NHP e *não* pelo HoTHP, plotando a intensidade aprendida pelos modelos
**treinados**.

**Sutileza conceitual (importante):** a bimodalidade do *tempo* não vive na intensidade
por tipo, e sim na **intensidade total** (*ground intensity*) `λ*(τ) = Σₖ λₖ(τ)`, em
função do tempo decorrido `τ` desde o último evento — porque é a `λ*` que governa **quando**
vem o próximo evento (de qualquer tipo), via a fórmula de sobrevivência
`f(τ) = λ*(τ)·exp(−∫₀^τ λ*)`. Por isso plotamos:

1. **Intensidade total `λ*(τ)`** — esperado: NHP **não-monótona** (cai e volta a subir),
   HoTHP **monótona**.
2. **Densidade implícita `f(τ)`** de cada modelo, sobreposta ao **histograma real** dos
   intervalos do Amazon — esperado: a `f` do NHP cobre os **dois** modos; a do HoTHP, um só.

**Checkpoints (Colab):** ao rodar no Colab, o notebook clona o repo, monta o Drive e
**descobre sozinho** os checkpoints de NHP e HoTHP do run final (`seed 42`) na mesma
estrutura do `HoTHP_validation_real_dataset` (`.../checkpoints/amazon__raw/<MODELO>/seed42/`).
Localmente, usa `../checkpoints/amazon/...`. Se algum não for achado, basta colar o caminho
em `NHP_CKPT`/`HOTHP_CKPT`.
"""))

cells.append(code(r"""# ── Setup Colab: clona o repo, instala deps, registra os modelos ─────────────
import os, sys, pathlib

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
REPO_DIR = 'ufc-easytpp'

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        os.system('git clone https://github.com/hugoramos/ufc-easytpp.git')
    else:
        os.system('git -C ufc-easytpp reset --hard -q && git -C ufc-easytpp pull -q')
    os.system('pip install omegaconf datasets pyyaml matplotlib pandas -q')
    # __init__.py minimo (inclui NHP) — evita imports de modelos com deps extras
    with open(os.path.join(REPO_DIR, 'easy_tpp/model/__init__.py'), 'w') as f:
        f.write(
            "from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel\n"
            "from easy_tpp.model.torch_model.torch_nhp   import NHP   as TorchNHP\n"
            "from easy_tpp.model.torch_model.torch_thp   import THP   as TorchTHP\n"
            "from easy_tpp.model.torch_model.torch_rothp import RoTHP as TorchRoTHP\n"
            "from easy_tpp.model.torch_model.torch_hothp import HoTHP as TorchHoTHP\n")
    sys.path.insert(0, REPO_DIR)
print('IN_COLAB =', IN_COLAB)"""))

cells.append(code(r"""# ── Imports, arquitetura, Drive e descoberta dos checkpoints ─────────────────
import glob, tempfile, warnings, math
warnings.filterwarnings('ignore')
import numpy as np
import torch, yaml
import matplotlib.pyplot as plt
from datasets import load_dataset

if not IN_COLAB:   # garante o repo no path quando rodando localmente (ex.: a partir de notebooks/)
    _root = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
    if _root not in sys.path:
        sys.path.insert(0, _root)

import easy_tpp.model  # noqa  (registra as subclasses p/ generate_model_from_config)
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.config_factory import Config

torch.manual_seed(0)
DEVICE = 'cpu'

# arquitetura do run da dissertação (HIDDEN=64, HEADS=2, LAYERS=2, TIME_EMB=16, use_ln=False)
ARCH = dict(hidden_size=64, num_heads=2, num_layers=2, time_emb_size=16,
            dropout=0.1, use_ln=False)
NUM_EVENT_TYPES = 16
PAD_TOKEN_ID    = 16
SEED            = 42          # run final: seeds 42/142/242 — escolha qual usar

# Onde estão os checkpoints?  No Colab: Drive (mesma estrutura do HoTHP_validation_real_dataset).
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CKPT_DIR = pathlib.Path('/content/drive/MyDrive/HoTHP_validation_real_dataset/checkpoints')
else:
    CKPT_DIR = pathlib.Path('../checkpoints') if os.path.basename(os.getcwd()) == 'notebooks' \
               else pathlib.Path('checkpoints')

def achar_ckpt(model_id):
    # estrutura do Drive (run final): checkpoints/amazon__raw/<MODELO>/seed42/<uid>/models/saved_model
    pats = [str(CKPT_DIR / 'amazon__raw' / model_id / f'seed{SEED}' / '*' / 'models' / 'saved_model'),
            # estrutura local antiga: checkpoints/amazon/<MODELO>/seed2019/<uid>/models/saved_model
            str(CKPT_DIR / 'amazon' / model_id / '*' / '*' / 'models' / 'saved_model')]
    for p in pats:
        cands = glob.glob(p)
        if cands:
            return sorted(cands, key=os.path.getmtime)[-1]
    return None

HOTHP_CKPT = achar_ckpt('HoTHP')
NHP_CKPT   = achar_ckpt('NHP')   # auto-descoberto no Drive; ou cole o caminho manualmente

print('CKPT_DIR  :', CKPT_DIR)
print('HoTHP ckpt:', HOTHP_CKPT or '(não encontrado — confira CKPT_DIR/SEED)')
print('NHP   ckpt:', NHP_CKPT   or '(não encontrado — cole NHP_CKPT manualmente)')"""))

cells.append(md(r"""## 1. Construir e carregar os modelos treinados

`build_model` monta a config mínima (mesma maquinaria do runner), instancia o modelo via
`generate_model_from_config` e carrega os pesos do `saved_model` (`load_state_dict`)."""))

cells.append(code(r"""def make_config(model_id, stage='eval'):
    exp_id = f'{model_id}_{stage}'
    model_cfg = {
        'hidden_size': ARCH['hidden_size'], 'num_heads': ARCH['num_heads'],
        'num_layers': ARCH['num_layers'], 'dropout': ARCH['dropout'],
        'time_emb_size': ARCH['time_emb_size'], 'use_ln': ARCH['use_ln'],
        'thinning': {'num_sample': 1, 'num_exp': 50, 'look_ahead_time': 10,
                     'patience_counter': 5, 'over_sample_rate': 5,
                     'num_samples_boundary': 5, 'dtime_max': 10, 'num_seq': 10, 'num_step_gen': 1},
    }
    cfg = {
        'pipeline_config_id': 'runner_config',
        'data': {'amazon': {'data_format': 'json', 'train_dir': 'x', 'valid_dir': 'x', 'test_dir': 'x',
                            'data_specs': {'num_event_types': NUM_EVENT_TYPES, 'pad_token_id': PAD_TOKEN_ID,
                                           'padding_side': 'right', 'truncation_side': 'right',
                                           'max_len': 128}}},
        exp_id: {'base_config': {'stage': stage, 'backend': 'torch', 'dataset_id': 'amazon',
                                 'runner_id': 'std_tpp', 'model_id': model_id,
                                 'base_dir': './_tmp/'},
                 'trainer_config': {'batch_size': 1, 'max_epoch': 1, 'seed': 0, 'gpu': -1,
                                    'metrics': ['acc', 'rmse']},
                 'model_config': model_cfg},
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f); tmp = f.name
    try:
        return Config.build_from_yaml_file(tmp, experiment_id=exp_id)
    finally:
        os.unlink(tmp)

def build_model(model_id, ckpt_path):
    config = make_config(model_id)
    model = TorchBaseModel.generate_model_from_config(config.model_config)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'  {model_id}: carregado | faltando={len(missing)} inesperado={len(unexpected)}')
    model.to(DEVICE).eval()
    return model

models = {}
models['HoTHP'] = build_model('HoTHP', HOTHP_CKPT)
if NHP_CKPT:
    models['NHP'] = build_model('NHP', NHP_CKPT)
print('modelos prontos:', list(models))"""))

cells.append(md(r"""## 2. Carregar sequências reais do Amazon (tensores)

Pegamos algumas sequências do split de teste e montamos os tensores que os modelos esperam:
`time_seqs` (timestamps), `time_delta_seqs` (gaps) e `type_seqs` (tipos)."""))

cells.append(code(r"""N_SEQS = 128   # quantas sequências usar para a média

ds = load_dataset('easytpp/amazon', split='test')
seqs = []
for row in ds:
    t  = np.asarray(row['time_since_start'], dtype=np.float64)
    ty = list(row['type_event'])
    if len(t) >= 4:
        seqs.append((t, ty))
    if len(seqs) >= N_SEQS:
        break

Lmax = max(len(t) for t, _ in seqs)
B = len(seqs)
time_seqs  = np.zeros((B, Lmax)); dtime_seqs = np.zeros((B, Lmax))
type_seqs  = np.full((B, Lmax), PAD_TOKEN_ID, dtype=np.int64); valid = np.zeros((B, Lmax), dtype=bool)
for i, (t, ty) in enumerate(seqs):
    L = len(t)
    time_seqs[i, :L]  = t
    dtime_seqs[i, 1:L] = np.diff(t)
    type_seqs[i, :L]  = ty
    valid[i, :L]      = True

time_seqs  = torch.tensor(time_seqs,  dtype=torch.float32, device=DEVICE)
dtime_seqs = torch.tensor(dtime_seqs, dtype=torch.float32, device=DEVICE)
type_seqs  = torch.tensor(type_seqs,  dtype=torch.long,    device=DEVICE)
print(f'{B} sequências, comprimento máx {Lmax}')"""))

cells.append(md(r"""## 3. Extrair a intensidade total `λ*(τ)`

Para cada posição de evento `j`, avaliamos a intensidade de cada tipo num grid de tempos
decorridos `τ ∈ [0, T]` *depois* de `t_j`, e somamos sobre os tipos → `λ*(τ)`. Depois
tiramos a **média sobre todas as posições válidas e sequências** para obter a curva típica."""))

cells.append(code(r"""T_MAX   = 1.0           # cobre os dois modos do Amazon (~0.013 e ~0.75)
N_GRID  = 200
tau = torch.linspace(0, T_MAX, N_GRID, device=DEVICE)

@torch.no_grad()
def ground_intensity(model):
    # sample_dtimes [B, L, N_GRID] = mesmo grid tau em toda posição
    sample_dtimes = tau[None, None, :].expand(B, Lmax, N_GRID).contiguous()
    lamk = model.compute_intensities_at_sample_times(time_seqs, dtime_seqs, type_seqs, sample_dtimes)
    lam_star = lamk.sum(dim=-1)                      # [B, L, N_GRID] soma sobre tipos
    # média sobre posições válidas que têm um próximo evento (j = 0 .. L-2)
    mask = torch.zeros(B, Lmax, dtype=torch.bool, device=DEVICE)
    for i, (t, _) in enumerate(seqs):
        mask[i, :len(t) - 1] = True
    sel = lam_star[mask]                              # [n_pos, N_GRID]
    return sel.mean(0).cpu().numpy(), sel.cpu().numpy()

curvas = {}
for name, m in models.items():
    media, todas = ground_intensity(m)
    curvas[name] = (media, todas)
    print(f'{name}: λ*(τ) extraída — shape {todas.shape}')"""))

cells.append(md(r"""## 4. Gráfico 1 — Intensidade total `λ*(τ)`

A prova visual da forma: NHP deve **cair e voltar a subir** (não-monótona); HoTHP deve ser
**monótona** (só decai ou só sobe)."""))

cells.append(code(r"""taun = tau.cpu().numpy()
COR = {'NHP': '#2ca02c', 'HoTHP': '#d62728', 'THP': '#d62728'}
fig, ax = plt.subplots(1, len(curvas), figsize=(6 * len(curvas), 4), squeeze=False)
for k, (name, (media, todas)) in enumerate(curvas.items()):
    a = ax[0][k]
    idx = np.random.default_rng(0).choice(len(todas), size=min(40, len(todas)), replace=False)
    for j in idx:
        a.plot(taun, todas[j], color=COR.get(name, 'gray'), alpha=0.06, lw=1)
    a.plot(taun, media, color=COR.get(name, 'black'), lw=3, label='média')
    a.set_title(f'{name} — intensidade total  λ*(τ)', fontweight='bold')
    a.set_xlabel('τ = tempo desde o último evento'); a.set_ylabel('λ*(τ)')
    a.legend(); a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig('fig_intensity_real_amazon.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(md(r"""## 5. Gráfico 2 — Densidade implícita vs histograma real (a prova dos nove)

De cada `λ*(τ)` derivamos a densidade do próximo intervalo `f(τ)=λ*·exp(−∫λ*)` e a
sobrepomos ao histograma real dos intervalos do Amazon. Se o NHP estiver de fato tratando
a bimodalidade, a curva verde deve subir **nos dois** montes; a vermelha (HoTHP), num só."""))

cells.append(code(r"""# histograma real dos gaps do Amazon (mesmo split de teste)
gaps_reais = np.concatenate([np.diff(t) for t, _ in seqs])
gaps_reais = gaps_reais[gaps_reais <= T_MAX]

def densidade_implicita(todas):
    # f(τ) = λ*(τ) · exp(−∫₀^τ λ*),  por (seq,pos), depois média
    dt = taun[1] - taun[0]
    Lam = np.cumsum((todas[:, 1:] + todas[:, :-1]) / 2 * dt, axis=1)
    Lam = np.concatenate([np.zeros((len(todas), 1)), Lam], axis=1)
    f = todas * np.exp(-Lam)
    return f.mean(0)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(gaps_reais, bins=70, range=(0, T_MAX), density=True, color='0.8',
        edgecolor='none', label='intervalos reais (Amazon)')
for name, (media, todas) in curvas.items():
    f = densidade_implicita(todas)
    area = np.trapz(f, taun)
    ax.plot(taun, f / area, color=COR.get(name, 'black'), lw=2.5, label=f'densidade implícita — {name}')
ax.set_xlabel('Δt = intervalo até o próximo evento'); ax.set_ylabel('densidade')
ax.set_title('Densidade implícita de cada modelo vs dados reais', fontweight='bold')
ax.legend(); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig('fig_density_real_amazon.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(md(r"""## 6. Leitura

- **Gráfico 1:** se a `λ*(τ)` do NHP **cai e volta a subir** e a do HoTHP é **monótona**,
  está confirmada empiricamente — com os pesos treinados — a tese de que só o NHP produz
  intensidade não-monótona.
- **Gráfico 2:** se a densidade implícita do NHP acompanha os **dois** montes do histograma
  real e a do HoTHP fica presa em **um**, está dada a prova dos nove: o NHP trata a
  bimodalidade do Amazon; a família THP (HoTHP incluído) não.

Tudo a partir dos modelos treinados e dos dados reais, reproduzível.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

out = pathlib.Path(__file__).parent / "HoTHP_vs_NHP_Intensity_Amazon.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("escrito:", out)
