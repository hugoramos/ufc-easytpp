# 📚 EasyTPP Notebooks Index

This directory contains notebooks and guides for working with EasyTPP.

---

## 🆕 NEW: THP Model - Minimal Official Implementation

**Just created for you!** Complete minimal setup for THP model training.

| File | Description | Use When |
|------|-------------|----------|
| **`THP_minimal_stackoverflow.ipynb`** ⭐ | **START HERE!** Minimal code to train THP on StackOverflow dataset from HuggingFace | You want to train THP with minimal code |
| `CHEATSHEET.md` | Quick reference card - absolute minimal code | You need a quick reminder |
| `THP_QUICKSTART.md` | Answers to your questions about datasets | You want detailed explanations |
| `README_THP.md` | Complete documentation and guide | You want comprehensive info |
| `check_dataset_info.py` | Helper script to inspect any dataset | You need to find num_event_types |

**Quick Start:**
```bash
jupyter notebook THP_minimal_stackoverflow.ipynb
# Run all cells → Get trained THP model!
```

---

## 📖 Official EasyTPP Tutorials

Original notebooks from the EasyTPP library:

| File | Description |
|------|-------------|
| `easytpp_1_dataset.ipynb` | Dataset loading and preprocessing |
| `easytpp_2_tfb_wb.ipynb` | TensorBoard and Weights & Biases integration |
| `easytpp_3_train_eval.ipynb` | Complete training and evaluation pipeline |

---

## 🎓 Architecture Understanding (Portuguese)

| File | Description |
|------|-------------|
| `understanding_easytpp_architecture.ipynb` | Deep dive into EasyTPP architecture (NHP, THP, etc.) |
| `README_tutorial.md` | Guide for the architecture notebook |

---

## 🏥 Domain-Specific Notebooks

| File | Description |
|------|-------------|
| `s2p2_preprocess_ehrshot_cpt4.ipynb` | Preprocessing for EHR data (CPT-4 codes) |

---

## 🗂️ Quick Navigation

### I want to...

**Train THP model immediately:**
→ Open `THP_minimal_stackoverflow.ipynb`

**Understand how EasyTPP works internally:**
→ Open `understanding_easytpp_architecture.ipynb`

**Learn the complete EasyTPP pipeline:**
→ Start with `easytpp_1_dataset.ipynb`, then `easytpp_3_train_eval.ipynb`

**Switch to a different dataset (taxi, retweet, etc.):**
→ Run `python check_dataset_info.py easytpp/DATASET_NAME`

**Get quick code snippet:**
→ Open `CHEATSHEET.md`

**Understand dataset loading options:**
→ Read `THP_QUICKSTART.md`

---

## 📊 Dataset Loading Options

### Option 1: HuggingFace (Recommended ✅)

```python
data:
  stackoverflow:
    data_format: json
    train_dir: easytpp/stackoverflow
    valid_dir: easytpp/stackoverflow
    test_dir: easytpp/stackoverflow
```

**Advantages:**
- ✅ No manual downloads
- ✅ Always up-to-date
- ✅ Official datasets
- ✅ Future-proof (pickle being deprecated)

### Option 2: Local files (Legacy)

```python
data:
  stackoverflow:
    data_format: pkl
    train_dir: ./data/stackoverflow/train.pkl
    valid_dir: ./data/stackoverflow/dev.pkl
    test_dir: ./data/stackoverflow/test.pkl
```

**Use only if:** You already have pickle files and can't use internet.

---

## 🎯 Your Research Path

### Phase 1: Understand the Baseline
1. ✅ Run `THP_minimal_stackoverflow.ipynb` to get baseline THP results
2. ✅ Read `understanding_easytpp_architecture.ipynb` to understand internals
3. ✅ Review `easytpp_3_train_eval.ipynb` for complete pipeline

### Phase 2: Experiment with Datasets
1. Use `check_dataset_info.py` to explore available datasets
2. Modify `THP_minimal_stackoverflow.ipynb` to test different datasets
3. Compare results across datasets

### Phase 3: Create Your Variation
1. Copy `THP_minimal_stackoverflow.ipynb` → `THP_my_variation.ipynb`
2. Modify the model architecture (add your innovations)
3. Train and compare against baseline

---

## 📦 Available Datasets

All accessible via `easytpp/DATASET_NAME`:

| Dataset | HuggingFace Path | Event Types | Domain |
|---------|------------------|-------------|--------|
| StackOverflow | `easytpp/stackoverflow` | 22 | Q&A platform |
| Taxi | `easytpp/taxi` | 10 | NYC taxi pickups |
| Retweet | `easytpp/retweet` | ? | Twitter retweets |
| Earthquake | `easytpp/earthquake` | ? | Seismic events |
| Amazon | `easytpp/amazon` | ? | Product reviews |

To find event types: `python check_dataset_info.py easytpp/DATASET_NAME`

---

## 🚀 Minimal Code to Train THP

From `THP_minimal_stackoverflow.ipynb`:

```python
# 1. Config (in notebook cell)
yaml_content = """
pipeline_config_id: runner_config
data:
  stackoverflow:
    data_format: json
    train_dir: easytpp/stackoverflow
    valid_dir: easytpp/stackoverflow
    test_dir: easytpp/stackoverflow
    data_specs:
      num_event_types: 22
THP_train:
  base_config:
    stage: train
    model_id: THP
  trainer_config:
    max_epoch: 10
"""
with open("config.yaml", "w") as f:
    f.write(yaml_content)

# 2. Train (3 lines!)
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
Runner.build_from_config(
    Config.build_from_yaml_file('./config.yaml', experiment_id='THP_train')
).run()
```

That's it! ✓

---

## 🔧 Helper Tools

### Dataset Inspector
```bash
python check_dataset_info.py easytpp/stackoverflow
```

Outputs:
- Dataset statistics
- Number of event types
- Suggested configuration values
- Ready-to-use YAML template

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `INDEX.md` | This file - navigation hub |
| `README_THP.md` | Complete THP guide |
| `THP_QUICKSTART.md` | Quick answers and guide |
| `CHEATSHEET.md` | One-page reference |
| `README_tutorial.md` | Architecture tutorial guide (PT) |

---

## 💡 Pro Tips

1. **Always start with HuggingFace datasets** - They're official and maintained
2. **Use check_dataset_info.py** - Saves time finding correct parameters
3. **Keep THP_minimal as reference** - Don't modify it, copy for experiments
4. **Compare same metrics** - Use same dataset/config for fair comparison
5. **Start small** - 10 epochs first, then increase if needed

---

## 🎓 Learning Path

**Beginner:**
1. Run `THP_minimal_stackoverflow.ipynb` (5 min)
2. Read `CHEATSHEET.md` (2 min)
3. Experiment with different datasets (10 min)

**Intermediate:**
1. Read `THP_QUICKSTART.md` (10 min)
2. Study `easytpp_3_train_eval.ipynb` (30 min)
3. Modify hyperparameters and observe results (1 hour)

**Advanced:**
1. Study `understanding_easytpp_architecture.ipynb` (1 hour)
2. Read `README_THP.md` completely (20 min)
3. Create your own model variation (ongoing)

---

## ✅ Summary

**For immediate THP training:**
```bash
jupyter notebook THP_minimal_stackoverflow.ipynb
# Run all cells
```

**For understanding the architecture:**
```bash
jupyter notebook understanding_easytpp_architecture.ipynb
```

**For dataset information:**
```bash
python check_dataset_info.py easytpp/YOUR_DATASET
```

---

**Last Updated:** November 20, 2025  
**Location:** `/Users/hugoramossoares/Sites/EasyTemporalPointProcess/notebooks/`

---

## 🎯 Your Questions - Answered

✅ **Is HuggingFace the best way?** → YES! See `THP_QUICKSTART.md`  
✅ **Easy to test other datasets?** → YES! Change 3 lines, see `CHEATSHEET.md`  
✅ **Is it official?** → Very likely YES, see `README_THP.md`

**Ready to start?** Open `THP_minimal_stackoverflow.ipynb` 🚀

