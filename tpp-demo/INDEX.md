# TPP-DEMO Index

## 🗂️ File Directory

### 🎯 Main Files (Use These!)

| File | Purpose | When to Use |
|------|---------|-------------|
| **TPP_Model_Comparison.ipynb** | Interactive notebook | **Primary tool** - Use this for model comparison |
| **model_comparison.py** | Python script | When you prefer command-line or batch processing |
| **example_usage.py** | Usage examples | After training, to use/analyze models |

### 📚 Documentation (Read These!)

| File | Content | Read Time | Priority |
|------|---------|-----------|----------|
| **START_HERE.md** | Quick orientation | 2 min | 🔴 High |
| **QUICKSTART.md** | 3-step guide | 3 min | 🔴 High |
| **OVERVIEW.md** | System overview | 10 min | 🟡 Medium |
| **README.md** | Full reference | 15 min | 🟢 Low |
| **INDEX.md** (this file) | File directory | 1 min | 🟡 Medium |

### 📁 Directories

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `configs/` | YAML config files | Model configurations |
| `data/taxi/` | train/dev/test.pkl | Your dataset |
| `checkpoints/` | Trained models | Saved model weights |
| `results/` | CSV & PNG files | Comparison results |
| `scripts/` | Utility scripts | Data processing tools |

---

## 🎯 Quick Navigation

### I want to...

**→ Run my first experiment**
1. Read: `START_HERE.md`
2. Open: `TPP_Model_Comparison.ipynb`
3. Run all cells

**→ Understand the system**
1. Read: `OVERVIEW.md`
2. Browse: `README.md`

**→ Get results quickly**
1. Read: `QUICKSTART.md`
2. Run: `jupyter notebook TPP_Model_Comparison.ipynb`

**→ Use a trained model**
1. Check: `checkpoints/` directory
2. Read: `example_usage.py`
3. Adapt examples to your needs

**→ Understand the code**
1. Open: `model_comparison.py`
2. Read: Function docstrings
3. Refer to: `README.md` for details

**→ Troubleshoot issues**
1. Check: `README.md` → Troubleshooting section
2. Check: `QUICKSTART.md` → Problems section
3. Review: Error messages in notebook

---

## 📊 Workflow Guide

### Beginner Workflow
```
START_HERE.md → QUICKSTART.md → TPP_Model_Comparison.ipynb
```

### Advanced Workflow
```
OVERVIEW.md → README.md → model_comparison.py → example_usage.py
```

### Quick Test Workflow
```
QUICKSTART.md → TPP_Model_Comparison.ipynb (2 models, 5 epochs)
```

### Production Workflow
```
README.md → TPP_Model_Comparison.ipynb (all models, 50+ epochs) → example_usage.py
```

---

## 🔍 Find Information By Topic

### Configuration
- **Basic**: `QUICKSTART.md` → Configuration section
- **Detailed**: `README.md` → Configuration section
- **Examples**: `OVERVIEW.md` → Common Configurations

### Models
- **List**: `START_HERE.md` or any doc file
- **Details**: `README.md` → Available Models
- **Papers**: `README.md` → References

### Training
- **Quick**: `QUICKSTART.md` → Step 3
- **Detailed**: `README.md` → Training section
- **Code**: `model_comparison.py` → train_model()

### Evaluation
- **Quick**: `QUICKSTART.md` → What You'll Get
- **Detailed**: `README.md` → Output section
- **Code**: `model_comparison.py` → evaluate_model()

### Results
- **Understanding**: `OVERVIEW.md` → Understanding the Output
- **Location**: `results/` directory
- **Format**: CSV tables + PNG plots

### Troubleshooting
- **Common issues**: `QUICKSTART.md` → Problems
- **Detailed help**: `README.md` → Troubleshooting
- **Examples**: `OVERVIEW.md` → Troubleshooting

---

## 📈 Complexity Levels

### Level 1: Beginner
**Goal**: Run first experiment
- Read: `START_HERE.md`, `QUICKSTART.md`
- Use: `TPP_Model_Comparison.ipynb`
- Config: 2 models, 5 epochs

### Level 2: Intermediate
**Goal**: Compare multiple models
- Read: `OVERVIEW.md`
- Use: `TPP_Model_Comparison.ipynb`
- Config: 4-6 models, 10-20 epochs

### Level 3: Advanced
**Goal**: Optimize and deploy
- Read: `README.md`, `example_usage.py`
- Use: Both notebook and scripts
- Config: All models, 50+ epochs, hyperparameter tuning

---

## 🎓 Learning Path

### Day 1: Getting Started
1. ✅ Read `START_HERE.md` (2 min)
2. ✅ Read `QUICKSTART.md` (3 min)
3. ✅ Run notebook with 2 models, 5 epochs (10 min)
4. ✅ Inspect results (5 min)

**Total: ~20 minutes**

### Day 2: Full Comparison
1. ✅ Read `OVERVIEW.md` (10 min)
2. ✅ Run notebook with 4 models, 10 epochs (60 min)
3. ✅ Analyze results (15 min)

**Total: ~85 minutes**

### Day 3: Deep Dive
1. ✅ Read `README.md` (15 min)
2. ✅ Study `model_comparison.py` (20 min)
3. ✅ Experiment with hyperparameters (120 min)

**Total: ~155 minutes**

### Day 4: Production
1. ✅ Train best model with optimal settings (180 min)
2. ✅ Use `example_usage.py` for predictions (30 min)
3. ✅ Document findings (30 min)

**Total: ~240 minutes**

---

## 🔗 External Resources

- **EasyTPP Documentation**: https://ant-research.github.io/EasyTemporalPointProcess/
- **EasyTPP GitHub**: https://github.com/ant-research/EasyTemporalPointProcess
- **Jupyter Documentation**: https://jupyter.org/documentation

---

## ✅ Checklist

### Before Starting
- [ ] Python 3.7+ installed
- [ ] Jupyter installed: `pip install jupyter`
- [ ] EasyTPP installed: `cd .. && pip install -e .`
- [ ] In correct directory: `cd tpp-demo`
- [ ] Data files present: `ls data/taxi/*.pkl`

### First Run
- [ ] Read `START_HERE.md`
- [ ] Read `QUICKSTART.md`
- [ ] Open notebook: `jupyter notebook TPP_Model_Comparison.ipynb`
- [ ] Configure: Set models and epochs
- [ ] Run: Execute all cells
- [ ] Check: Review results in `results/`

### After Results
- [ ] Review comparison table
- [ ] Check visualization plots
- [ ] Identify best model
- [ ] Note checkpoint location
- [ ] Plan next experiments

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Open notebook | `jupyter notebook TPP_Model_Comparison.ipynb` |
| Run script | `python model_comparison.py` |
| Check results | `ls results/` |
| View checkpoints | `ls checkpoints/` |
| Install EasyTPP | `cd .. && pip install -e .` |
| List data | `ls data/taxi/` |

---

**Last Updated**: November 11, 2025  
**Version**: 1.0  
**Status**: ✅ Complete and Ready to Use

