"""Gera o notebook Dataset_Profile_Validation.ipynb (perfis dos datasets EasyTPP)."""
import json, pathlib

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}

cells = []

cells.append(md(r"""# Validação dos perfis dos datasets — EasyTPP

Notebook **auto-contido e auditável**. Ele baixa os datasets reais do EasyTPP direto do
HuggingFace e calcula o "perfil temporal" de cada um **do zero**, usando apenas
`numpy`, `pandas` e `matplotlib` — nenhuma função do projeto, nenhuma caixa-preta.

O intervalo entre eventos é computado como `Δt = diff(time_since_start)` (a diferença
crua dos *timestamps*), e cada passo é validado contra o campo `time_since_last_event`
fornecido pelo dataset.

Objetivo: confirmar, com código que qualquer um pode reproduzir, que:
1. **Amazon** tem intervalos **bimodais com um vazio** (dois "relógios": ~0.013 e ~0.75);
2. **StackOverflow / Taxi / etc.** têm intervalos **unimodais decaindo** (cara de Hawkes);
3. As **marcas** (tipos de evento) são quase imprevisíveis em alguns datasets (ex.: Amazon)
   e muito previsíveis em outros (ex.: Taxi).
"""))

cells.append(code(r"""import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

pd.set_option('display.float_format', lambda v: f'{v:.4f}')

# Todos os datasets do org easytpp/ que carregam pelo HuggingFace:
DATASETS = ['amazon', 'taxi', 'taobao', 'stackoverflow', 'retweet', 'earthquake', 'volcano']
print('datasets a analisar:', DATASETS)"""))

cells.append(md(r"""## 1. Download e extração dos intervalos (a partir dos timestamps crus)

Para cada sequência: `Δt = diff(time_since_start)`. Conferimos que esse cálculo
reproduz exatamente o campo `time_since_last_event` que vem no dataset
(coluna `gaps == diff?` na tabela mais abaixo)."""))

cells.append(code(r"""def carregar(name):
    ds = load_dataset(f'easytpp/{name}', split='train')
    gaps, types, lengths = [], [], []
    bate_com_campo = True
    for row in ds:
        t  = np.asarray(row['time_since_start'], dtype=float)
        ty = list(row['type_event'])
        lengths.append(len(t))
        if len(t) < 2:
            continue
        g = np.diff(t)                                  # intervalo = diferença dos timestamps
        prov = np.asarray(row['time_since_last_event'], dtype=float)[1:]
        if not np.allclose(g, prov, atol=1e-6):         # validação do nosso cálculo
            bate_com_campo = False
        gaps.extend(g.tolist())
        types.extend(ty)
    return np.asarray(gaps), np.asarray(types), np.asarray(lengths), bate_com_campo

DATA = {}
for name in DATASETS:
    print('baixando', name, '...')
    DATA[name] = carregar(name)
print('ok')"""))

cells.append(md(r"""## 2. Tabela-resumo do perfil temporal

`CV` = coeficiente de variação (`std/média`) dos intervalos. É o primeiro sinal:
distribuições com um pico só e cauda longa têm `CV` alto; a estrutura bimodal do Amazon
dá um `CV` baixo (~0.66) porque os dois picos são estreitos."""))

cells.append(code(r"""linhas = []
for name, (g, ty, L, ok) in DATA.items():
    linhas.append({
        'dataset':    name,
        'n_seqs':     len(L),
        'len_med':    int(np.median(L)),
        'len_max':    int(L.max()),
        'n_tipos':    int(len(np.unique(ty))),
        'gap_medio':  g.mean(),
        'gap_std':    g.std(),
        'CV':         g.std() / g.mean(),
        'gap_p50':    np.percentile(g, 50),
        'gap_p99':    np.percentile(g, 99),
        'gap_max':    g.max(),
        'gaps==diff?': ok,
    })
resumo = pd.DataFrame(linhas).set_index('dataset')
resumo"""))

cells.append(md(r"""## 3. Histogramas dos intervalos entre eventos

A forma fala por si. **Amazon**: duas barras isoladas com um vazio gritante no meio.
Os demais: uma única massa decaindo a partir do zero (perfil clássico de auto-excitação).
Cortamos 1% de cauda (`p99`) só para enxergar a forma — a fração visível está no título."""))

cells.append(code(r"""n = len(DATASETS); ncol = 3; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
axes = axes.ravel()
for ax, name in zip(axes, DATASETS):
    g = DATA[name][0]
    hi = np.percentile(g, 99)
    ax.hist(g, bins=60, range=(0, hi), color='steelblue', edgecolor='white', linewidth=0.3)
    vis = (g <= hi).mean() * 100
    cv = g.std() / g.mean()
    ax.set_title(f'{name}   (CV={cv:.2f}, {vis:.0f}% visível, máx={g.max():.1f})')
    ax.set_xlabel(r'intervalo entre eventos  $\Delta t$')
    ax.set_ylabel('contagem')
    ax.spines[['top', 'right']].set_visible(False)
for ax in axes[n:]:
    ax.axis('off')
fig.tight_layout()
fig.savefig('dataset_profiles_hist.png', dpi=150, bbox_inches='tight')
plt.show()
print('figura salva: dataset_profiles_hist.png')"""))

cells.append(md(r"""## 4. Detector de "vazio" (bimodalidade), sem achismo

Discretiza os intervalos em 40 faixas, identifica as faixas **populadas** (>0.2% da massa)
e mede quantas faixas **vazias** existem *entre* a primeira e a última populada.
Um `vazio` alto = distribuição com buraco no meio = bimodal. Esperado: Amazon alto, resto ~0."""))

cells.append(code(r"""def score_vazio(g, nbins=40, q=99.5):
    hi = np.percentile(g, q)
    c, _ = np.histogram(g[g <= hi], bins=nbins, range=(0, hi))
    limiar = 0.002 * c.sum()
    pop = np.where(c > limiar)[0]
    if len(pop) < 2:
        return 0.0, 0
    interior = c[pop[0]:pop[-1] + 1]
    vazias = int((interior <= limiar).sum())
    return vazias / len(interior), vazias

linhas = []
for name, (g, *_ ) in DATA.items():
    frac, n_vazias = score_vazio(g)
    linhas.append({'dataset': name, 'frac_faixas_vazias_no_meio': frac, 'n_faixas_vazias': n_vazias})
pd.DataFrame(linhas).set_index('dataset')"""))

cells.append(md(r"""## 5. As marcas (tipos) são previsíveis pelo histórico?

Compara a **entropia marginal** dos tipos (sortear pela frequência) com a **NLL de um
Markov de ordem 1** (prever o tipo a partir do tipo anterior). Se as duas forem quase
iguais, o histórico não ajuda → marcas ~ i.i.d. (caso do Amazon). Se a NLL do Markov for
bem menor, há forte estrutura sequencial (caso do Taxi)."""))

cells.append(code(r"""def previsibilidade_marcas(name):
    ds = load_dataset(f'easytpp/{name}', split='train')
    seqs = [list(r['type_event']) for r in ds]
    c = Counter(x for s in seqs for x in s); tot = sum(c.values())
    H0 = -sum((v / tot) * math.log(v / tot) for v in c.values())     # entropia marginal
    trans = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            trans[a][b] += 1
    num = den = 0.0
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            row = trans[a]; num += -math.log(row[b] / sum(row.values())); den += 1
    return len(c), H0, num / den

linhas = []
for name in DATASETS:
    K, H0, H1 = previsibilidade_marcas(name)
    linhas.append({'dataset': name, 'K_tipos': K, 'log(K)': math.log(K),
                   'entropia_marginal': H0, 'NLL_markov1': H1, 'reducao_pelo_historico': H0 - H1})
pd.DataFrame(linhas).set_index('dataset')"""))

cells.append(md(r"""## 6. "Cara de Hawkes?" — clustering temporal

Hawkes é **auto-excitante**: eventos vêm em rajadas, então intervalos curtos tendem a ser
seguidos de intervalos curtos → **autocorrelação positiva** dos `Δt`. Medimos a correlação
lag-1 dos intervalos dentro de cada sequência. Perto de zero = sem clustering = não-Hawkes
(caso do Amazon)."""))

cells.append(code(r"""def autocorr_lag1(name):
    ds = load_dataset(f'easytpp/{name}', split='train')
    prev, cur = [], []
    for r in ds:
        g = np.diff(np.asarray(r['time_since_start'], dtype=float))
        if len(g) >= 2:
            prev.extend(g[:-1]); cur.extend(g[1:])
    prev, cur = np.asarray(prev), np.asarray(cur)
    return float(np.corrcoef(prev, cur)[0, 1])

linhas = [{'dataset': name, 'corr(dt_i, dt_{i-1})': autocorr_lag1(name)} for name in DATASETS]
pd.DataFrame(linhas).set_index('dataset')"""))

cells.append(md(r"""## 7. Validação-chave: o histórico prevê *quando* vem o próximo evento?

Esta é a checagem que descarta o "pêra com banana": será que a bimodalidade do Amazon é
**resolvida pelo histórico** (aí cada evento teria um modo definido) ou é **irredutível**
(o próximo intervalo continua incerto mesmo sabendo o passado)?

Discretizamos cada `Δt` em **terços** (curto/médio/longo) e medimos a entropia do terço do
**próximo** intervalo, condicionada a: nada (base), terço anterior, tipo atual, e ambos.
Se a entropia quase não cai, o histórico **não** prevê o tempo → a densidade preditiva de
cada evento é, ela mesma, multimodal (o que castiga modelos de *hazard* monótono)."""))

cells.append(code(r"""def entropia_condicional_tempo(name):
    ds = load_dataset(f'easytpp/{name}', split='train')
    seqs = []
    for r in ds:
        g = np.diff(np.asarray(r['time_since_start'], dtype=float))
        seqs.append((g, list(r['type_event'])))
    todos = np.concatenate([g for g, _ in seqs])
    q1, q2 = np.percentile(todos, [33.33, 66.67])
    tercil = lambda x: 0 if x <= q1 else (1 if x <= q2 else 2)

    def H(keyfn):
        cnt = defaultdict(Counter)
        for g, ty in seqs:
            for i in range(1, len(g)):
                cnt[keyfn(g, ty, i)][tercil(g[i])] += 1
        num = den = 0.0
        for _, dist in cnt.items():
            n = sum(dist.values())
            for c in dist.values():
                if c > 0:
                    num += -c * math.log(c / n)
            den += n
        return num / den

    base = H(lambda g, ty, i: 0)
    return {
        'dataset': name,
        'H_base':              base,
        'H|terco_anterior':    H(lambda g, ty, i: tercil(g[i - 1])),
        'H|tipo_atual':        H(lambda g, ty, i: ty[i]),
        'H|terco+tipo':        H(lambda g, ty, i: (tercil(g[i - 1]), ty[i])),
    }

linhas = [entropia_condicional_tempo(name) for name in DATASETS]
tab = pd.DataFrame(linhas).set_index('dataset')
tab['reducao_max'] = tab['H_base'] - tab[['H|terco_anterior', 'H|tipo_atual', 'H|terco+tipo']].min(axis=1)
tab"""))

cells.append(md(r"""## 8. Leitura dos resultados

- **Histogramas (§3) + vazio (§4):** o Amazon deve ser o único com faixas vazias no meio
  (bimodal). Os demais aparecem como uma massa única decaindo.
- **Marcas (§5):** no Amazon a redução pelo histórico é mínima (marcas ~ aleatórias); no
  Taxi a NLL do Markov despenca (tipos muito previsíveis).
- **Clustering (§6):** autocorrelação dos `Δt` perto de zero no Amazon → sem rajadas → não
  tem assinatura de auto-excitação (Hawkes).
- **Tempo condicional (§7):** se `reducao_max` for pequena no Amazon, o histórico não prevê
  *quando* vem o próximo evento → a bimodalidade é **irredutível**, validando que um modelo
  precisa de densidade preditiva multimodal por evento (o que o NHP faz e a família THP não).

Tudo acima é reproduzível a partir dos dados crus, sem depender de nenhuma análise externa.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# adiciona ids estáveis
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

out = pathlib.Path(__file__).parent / "Dataset_Profile_Validation.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("escrito:", out)
