# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **EasyTPP** (ICLR 2024), a PyTorch toolkit for Temporal Point Process (TPP) research. The fork extends the original library with custom models under active research: **RoTHP** (Rotary embeddings in THP), **HoTHP** (Hyperbolic/HoPE embeddings in THP), and **SmurfTHP** (variants thereof). The codebase is exclusively PyTorch — TensorFlow support was removed upstream.

## Installation

```bash
# From source (development mode)
pip install -e .

# Or from PyPI (upstream)
pip install easy-tpp
```

No `requirements.txt` exists in this fork — dependencies are inferred from `setup.py` which calls `parse_requirements('requirements.txt')`, so running from source requires creating one or installing known deps: `torch`, `omegaconf`, `optuna`, `datasets` (HuggingFace).

## Running Tests

```bash
# From the tests/ directory (required — tests use relative paths for data files)
cd tests
python -m unittest test_nhp.py
python -m unittest test_data_loader.py

# Run a single test method
cd tests
python -m unittest test_nhp.TestNeuralHawkesProcess.test_forward_pass
```

Tests use `tests/synthetic_data.json` as fixture data.

## Training a Model

Training is fully config-driven. The entry point pattern is:

```python
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

config = Config.build_from_yaml_file('path/to/config.yaml', experiment_id='THP_train')
Runner.build_from_config(config).run()
```

Or using the example runner scripts:
```bash
cd examples
python run_retweet.py
```

**Config structure** (`pipeline_config_id: runner_config`):
- `data.<dataset_id>`: data paths and `num_event_types`, `pad_token_id`
- `<ExperimentId>_train/eval/gen`: `base_config` (stage, model_id, dataset_id, backend), `trainer_config`, `model_config`

Datasets can be loaded from HuggingFace (recommended) using `data_format: json` and `train_dir: easytpp/<dataset_name>`, or from local `.pkl`/`.json` files.

Reference configs: `examples/configs/experiment_config.yaml`, `notebooks/configs/`.

## Architecture

```
easy_tpp/
├── config_factory/     # Config classes (Config, RunnerConfig, ModelConfig, DataConfig)
│                         Built via OmegaConf; Config.build_from_yaml_file() is the entry point
├── runner/             # Runner.build_from_config() dispatches to TPPRunner ('std_tpp')
│   ├── base_runner.py  # Registrable base class
│   └── tpp_runner.py   # Main training/eval/gen loop
├── model/torch_model/  # All model implementations inherit TorchBaseModel
│   ├── torch_basemodel.py   # Base class: handles embedding, device, event sampler
│   ├── torch_baselayer.py   # Shared layers: MultiHeadAttention, EncoderLayer, ScaledSoftplus
│   ├── torch_thp.py         # THP (Transformer Hawkes Process) — base for local variants
│   ├── torch_rothp*.py      # RoTHP variants (rotary positional embeddings for timestamps)
│   ├── torch_hothp.py       # HoTHP (HoPE/hyperbolic embeddings, relative temporal distances)
│   ├── torch_smurf.py       # SmurfTHP variant
│   └── torch_*.py           # Upstream models: NHP, SAHP, AttNHP, RMTPP, FullyNN, S2P2, ODETPP
├── preprocess/         # Dataset, EventTokenizer, DataCollator, DataLoader
└── utils/              # Logging, metrics, seed, device utils
```

**Model registration**: `TorchBaseModel.generate_model_from_config()` scans all subclasses and matches by class name to `model_id` in config. Adding a new model requires only subclassing `TorchBaseModel` and importing it (so Python registers the subclass) before calling the runner.

**Key data flow**: YAML config → `Config.build_from_yaml_file` → `RunnerConfig` → `TPPRunner.__init__` initializes model via `TorchBaseModel.generate_model_from_config` → `TorchModelWrapper` handles optimizer/scheduler → `TPPRunner.run()` executes train/eval/gen loop.

## Custom Research Models (This Fork)

Local additions in `easy_tpp/model/torch_model/`:
- `torch_rothp.py` — RoTHP: THP with rotary time embeddings applied to Q/K in attention
- `torch_rothp_simple.py`, `torch_rothp_decay.py`, `torch_rothp_hybrid.py` — RoTHP variants
- `torch_hothp.py` — HoTHP: HoPE-style hyperbolic attention kernel, computes relative temporal distances directly to avoid float overflow; `theta_prime` is a learnable parameter constrained > max(theta_i)
- `torch_imrothp.py` — IMRoTHP variant
- `torch_smurf.py` — SmurfTHP

Comparison experiment configs and results live in `easy_tpp/model/torch_model/study/comparison_results/`.

Synthetic datasets for stress-testing these models are in `datasets/synthetic_hothp_scenarios/` and `datasets/synthetic_hothp_stress_scenarios/` (train/validation/test splits in JSON format).

## Notebooks

`notebooks/` contains Jupyter notebooks for experiments and analysis. Key notebooks are tracked in `notebooks/INDEX.md`. The active research notebook is `notebooks/Extrapolation_and_Attention_Analysis.ipynb` (currently modified).
