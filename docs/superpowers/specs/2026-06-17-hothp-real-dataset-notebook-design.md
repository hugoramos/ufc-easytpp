# Design — `HoTHP_validation_real_dataset.ipynb`

**Data:** 2026-06-17
**Local de entrega:** `dissertacao/final-material/HoTHP_validation_real_dataset/HoTHP_validation_real_dataset.ipynb`
**Referência (análoga):** `dissertacao/final-material/HoTHP_validation_synthetic_dataset/HoTHP_validation_synthetic_dataset.ipynb`
**Base comprovada do motor:** `notebooks/Benchmark_Real_Datasets_Colab.ipynb` (gerou as tabelas reais)

## Objetivo

Notebook único, reproduzível e cadenciável (Colab) que valida HoTHP contra NHP/THP/RoTHP
em **datasets reais** do EasyTPP, sob o protocolo de extrapolação "train-short / test-long",
produzindo `metrics_cache.json`, `results.json`, checkpoints e figuras **análogas** às do
notebook sintético — para comparação direta no PDF da dissertação.

## Decisões (confirmadas com o usuário)

1. **Motor:** Runner do EasyTPP (`Config.build_from_yaml_file` → `Runner.build_from_config`
   → `runner.run()` / `runner._evaluate_model`). Mesmo caminho do `Benchmark_Real_Datasets_Colab.ipynb`.
2. **Protocolo:** truncamento via `truncation_strategy: longest_first` + `max_len`.
   Treina com `max_len = train_len`; avalia com `max_len = fator × train_len`.
   **Fatores = [1, 2, 5]**. Um fator é pulado se `fator × train_len > max_len_real` do dataset.
3. **Modelos:** NHP, THP, RoTHP, HoTHP (lista única, mesmas cores do sintético).
4. **Persistência:** Google Drive montado; toda saída sob uma pasta no Drive.
5. **Código do modelo:** aplicar os **mesmos patches do notebook sintético final**
   (`_normalize_timestamps` no-op = subtrai 1º timestamp; patch numérico de `_attention`
   com `masked_fill(-1e4)`). Consequência aceita: os números das Tabelas 5.1/5.2 serão
   **regenerados** e o usuário atualizará as tabelas no PDF.
6. **Datasets ativos por padrão:** Amazon, StackOverflow, Taxi, Retweet (raw+norm).
   Taobao (e earthquake) ficam definidos porém **comentados**, prontos para descomentar.

## Normalização (eixo `variant`)

No caminho do Runner os dados reais chegam **crus** (sem divisão pela média). Logo:
- **`raw`**: dataset do HuggingFace `easytpp/<name>` direto + `_normalize_timestamps` no-op.
  Para o Retweet cru isso reproduz o colapso numérico documentado (NLL explode).
- **`norm`** (só Retweet por padrão): dataset JSON local pré-normalizado, dividindo
  todos os timestamps de cada sequência pela média dos gaps `>0` e recomputando
  `time_since_last_event` (lógica de `notebooks/prepare_normalized_datasets.py`).
  Reproduz a coluna "Normalizado" da Tabela 5.2.

`variant` é uma lista por dataset: a maioria = `['raw']`; Retweet = `['raw', 'norm']`.
Qualquer dataset pode ganhar `'norm'` adicionando à sua lista.

## Comprimentos de treino (da metodologia da dissertação, não do benchmark improvisado)

| Dataset | num_event_types | train_len | 5× | max_len real | extrapola? |
|---|---|---|---|---|---|
| Amazon | 16 | 18 | 90 | ~94 | sim ([1,2,5]) |
| StackOverflow | 22 | 20 | 100 | ~101 | sim ([1,2,5]) |
| Taxi | 10 | 7 | 35 | ~38 | sim ([1,2,5]) |
| Retweet | 3 | 50 | 250 | ~264 | sim ([1,2,5]) |
| Taobao (comentado) | ~17 (confirmar) | 20 | 100 | (confirmar) | confirmar |
| Earthquake (comentado) | 7 | — | — | ~18 | não |

> Correção sobre o benchmark original: ele usou `taxi train_max_len=38` (sem extrapolação).
> A metodologia da dissertação usa Taxi train=7 / test≈38, então o Taxi **extrapola** aqui.
> Os `train_len` e `num_event_types` são confirmados pela célula de análise de escala antes do treino.

## Persistência sem sobrescrita

Estrutura sob a pasta do Drive (`BASE_DIR`):
```
BASE_DIR/
  checkpoints/{dataset}__{variant}/{model}/seed{seed}/...   # geridos pelo EasyTPP
  datasets_norm/{dataset}/{train,dev,test}.json             # variantes pré-normalizadas
  metrics_cache.json
  results.json
  hardware_info.json
  figures/*.png
```

`metrics_cache.json` com chave profunda:
```
metrics_cache[f"{dataset}__{variant}"][model][str(seed)][str(factor)] = {nll, rmse, acc}
```
- Cada combinação (dataset, variant, model, seed, factor) é uma chave isolada → rodar
  Amazon+Taobao hoje e StackOverflow amanhã **só acrescenta chaves**; nada é apagado.
- Skip de treino verifica o **checkpoint físico** no disco (resiliente a reset do Colab;
  re-treina e invalida evals daquele modelo se o checkpoint sumiu).
- Skip de eval verifica a chave `(dataset,variant,model,seed,factor)` no cache.
- Re-habilitar um dataset já rodado: tabelas/figuras são reconstruídas do cache (instantâneo).
- `results.json` = versão serializável/agregada, salva ao fim (com blindagem para
  reconstruir do cache caso a sessão tenha reiniciado).

## Toggle de datasets

Dict `DATASETS` com todas as entradas; o usuário comenta/descomenta (como no benchmark
original). Deixar um dataset já treinado descomentado é barato (skip via cache).

## Estrutura de células (≈ ordem)

1. **Setup (Colab):** clone/pull do repo; aplica os patches (`_normalize_timestamps` no-op,
   `_attention` numérico); `__init__.py` mínimo incluindo **NHP**; fix de import-all no runner;
   monta **Google Drive** e define `BASE_DIR`; cria subpastas.
2. **Imports + config global:** SEEDS=[2019,2020,2021], MODELS, hiperparâmetros reais
   (hidden=64, time_emb=16, layers=2, heads=2, dropout=0.1, lr=1e-3, max_epoch=100,
   batch=256, use_ln=False), EXTRAP_FACTORS=[1,2,5], CORES por modelo.
3. **`DATASETS`** (dict com toggle) + cálculo de fatores viáveis por dataset.
4. **Análise de escala temporal** (igual à célula 3 do benchmark): comprimentos,
   `num_event_types`, estatísticas de Δt — confirma specs antes de treinar.
5. **Preparação das variantes `norm`** (Retweet): gera JSON local pré-normalizado no Drive
   se ainda não existir.
6. **Helpers:** `make_yaml(dataset, variant, model, seed, max_len, stage, pretrained_dir)`
   (aponta `train/valid/test_dir` para HF ou para o JSON local conforme `variant`),
   `write_yaml_and_load`, `get_model_dir`, cache load/save, skip helpers, `free_gpu`.
7. **Treino:** loop (dataset, variant, model, seed) com `max_len=train_len`; skip por
   checkpoint físico; salva tempos de treino.
8. **Avaliação:** para cada modelo treinado, avalia em cada fator viável
   (`max_len=fator×train_len`) via `runner._evaluate_model`; grava no `metrics_cache.json`
   após cada item (`nll = -loglike`, `acc`, `rmse`).
9. **Reconstrução de `all_results`** a partir do cache + `results.json`.
10. **Tabelas-resumo** (NLL, RMSE, Accuracy) `média ± std` por dataset/variant/fator.
11. **Figuras análogas ao sintético** (estilo de **linhas com banda ±std**, não barras):
    NLL/RMSE/Acc vs fator de extrapolação (um subplot por dataset/variant), grid
    consolidado (3 métricas × N datasets), salvos em `figures/`.
12. **Tempo de treino + hardware** (tabela e gráfico, como no benchmark).

## Fora de escope (YAGNI)

- Tuning de hiperparâmetros (usa os fixos da dissertação).
- Wilcoxon entre modelos (o sintético faz com 10 seeds; aqui são 3 seeds → opcional,
  fica de fora por padrão; pode ser adicionado se o usuário quiser).
- Análise de intensidade aprendida (vive em outro notebook).

## Riscos / notas

- **Retweet é caro** (HoTHP norm ~97 min/seed na Tabela 5.3). Recomenda-se habilitá-lo por
  último na execução cadenciada.
- Retweet `raw` deve produzir NLL explosivo para HoTHP/RoTHP (resultado esperado/documentado);
  plots usam `nanmean`/`nanstd` e clipping visual para não quebrar.
- `num_event_types` do Taobao deve ser confirmado na célula de análise antes de treinar.
