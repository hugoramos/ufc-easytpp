# TPP-DEMO Overview

## 📋 What You Have

I've created a complete toolkit for comparing Temporal Point Process (TPP) models on your datasets. Here's what's included:

### 🎯 Main Files

1. **`TPP_Model_Comparison.ipynb`** (20 KB, 22 cells)
   - **Interactive Jupyter notebook** for model comparison
   - Step-by-step workflow with explanations
   - Includes data inspection, training, evaluation, and visualization
   - **This is your main tool!**

2. **`model_comparison.py`** (13 KB)
   - Standalone Python script with the same functionality
   - Run from command line: `python model_comparison.py`
   - Good for batch jobs or automated experiments

3. **`example_usage.py`** (9 KB)
   - Shows how to load and use trained models
   - Examples for prediction and model comparison
   - Useful after you've trained models

### 📚 Documentation

4. **`README.md`** (6 KB)
   - Complete documentation
   - Project structure, configuration, troubleshooting
   - Reference for all features

5. **`QUICKSTART.md`** (3 KB)
   - Get results in 3 steps
   - Perfect for first-time users
   - **Start here if you're in a hurry!**

6. **`OVERVIEW.md`** (this file)
   - High-level summary of everything

## 🚀 How to Use

### For First-Time Users

1. **Read**: `QUICKSTART.md` (2 minutes)
2. **Run**: `jupyter notebook TPP_Model_Comparison.ipynb`
3. **Configure**: Set models and parameters
4. **Execute**: Run all cells
5. **Analyze**: Check results and plots

### For Advanced Users

1. **Customize**: Edit configuration in notebook or script
2. **Experiment**: Try different hyperparameters
3. **Compare**: Run multiple experiments with different settings
4. **Analyze**: Use `example_usage.py` for deeper analysis

## 📊 What the Notebook Does

### Section 1: Setup (Cells 1-3)
- Import libraries
- Set up plotting style
- Verify everything works

### Section 2: Configuration (Cells 4-5)
- **You configure here!**
- Choose models: NHP, RMTPP, THP, SAHP, etc.
- Set hyperparameters: epochs, batch size, learning rate
- Specify dataset

### Section 3: Data Inspection (Cells 6-7)
- Load train/dev/test data
- Show dataset statistics
- Display sample sequences

### Section 4: Helper Functions (Cells 8-11)
- Define configuration creation
- Define training function
- Define evaluation function

### Section 5: Training (Cells 12-13)
- **Train all selected models**
- Progress displayed for each model
- Checkpoints saved automatically

### Section 6: Evaluation (Cells 14-15)
- **Evaluate on test set**
- Extract metrics
- Save results

### Section 7: Results (Cells 16-17)
- **Extract NLL scores**
- Create comparison table
- Save to CSV

### Section 8: Visualization (Cells 18-19)
- **Create bar charts**
- Compare train/valid/test NLL
- Save plots as PNG

### Section 9: Ranking (Cells 20-21)
- **Rank models by performance**
- Identify best model
- Show checkpoint location

### Section 10: Summary (Cell 22)
- Key findings
- Next steps
- Recommendations

## 🎓 Understanding the Output

### During Execution

You'll see:
```
============================================================
Training NHP on taxi dataset
============================================================

Epoch 1/10: loss=2.456, acc=0.678
Epoch 2/10: loss=2.234, acc=0.712
...
Epoch 10/10: loss=1.987, acc=0.789

NHP: success
```

### Final Results

**Comparison Table:**
```
Model    Train NLL  Valid NLL  Test NLL   Checkpoint
NHP      2.345      2.456      2.478      ./checkpoints/12345_timestamp/
RMTPP    2.234      2.367      2.389      ./checkpoints/67890_timestamp/
THP      2.123      2.245      2.267      ./checkpoints/11111_timestamp/
SAHP     2.089      2.198      2.215      ./checkpoints/22222_timestamp/
```

**Model Ranking:**
```
1. SAHP    - Test NLL: 2.215
2. THP     - Test NLL: 2.267
3. RMTPP   - Test NLL: 2.389
4. NHP     - Test NLL: 2.478

🏆 Best Model: SAHP (Test NLL: 2.215)
```

**Files Created:**
- `results/model_comparison_20251111_105900.csv` - Results table
- `results/model_comparison_20251111_105900.png` - Comparison plots
- `checkpoints/*/models/saved_model` - Trained model weights

## 🔧 Common Configurations

### Quick Test (5-10 minutes)
```python
MODELS = ['NHP', 'RMTPP']
MAX_EPOCHS = 5
BATCH_SIZE = 512
GPU = -1
```

### Standard Comparison (30-60 minutes)
```python
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP']
MAX_EPOCHS = 10
BATCH_SIZE = 256
GPU = -1
```

### Comprehensive Evaluation (2-4 hours)
```python
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP', 'FullyNN', 'ODETPP']
MAX_EPOCHS = 50
BATCH_SIZE = 256
GPU = 0  # Use GPU
```

### Publication Quality (overnight)
```python
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP', 'FullyNN', 'IntensityFree', 'ODETPP', 'AttNHP']
MAX_EPOCHS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 128
GPU = 0
```

## 📁 File Organization

```
tpp-demo/
├── TPP_Model_Comparison.ipynb  ← Main notebook (USE THIS!)
├── model_comparison.py          ← Script version
├── example_usage.py             ← Usage examples
├── QUICKSTART.md                ← Quick start guide
├── README.md                    ← Full documentation
├── OVERVIEW.md                  ← This file
│
├── configs/
│   └── taxi_nhp.yaml           ← Example config
│
├── data/
│   └── taxi/
│       ├── train.pkl           ← Your training data
│       ├── dev.pkl             ← Your validation data
│       └── test.pkl            ← Your test data
│
├── checkpoints/                ← Trained models saved here
│   └── <pid>_<timestamp>/
│       ├── models/
│       │   └── saved_model     ← Model weights
│       ├── log                 ← Training log
│       └── *_output.yaml       ← Metrics
│
├── results/                    ← Results saved here
│   ├── model_comparison_*.csv  ← Results table
│   └── model_comparison_*.png  ← Plots
│
└── scripts/                    ← Utility scripts
    ├── peek_pkl.py
    ├── inspect_pkl.py
    └── ...
```

## 🎯 Typical Workflow

### Day 1: Initial Exploration
1. Run notebook with 2-3 models, 5 epochs
2. Check if everything works
3. Inspect results

### Day 2: Full Comparison
1. Run notebook with 4-6 models, 10-20 epochs
2. Identify top 2-3 models
3. Analyze results

### Day 3: Fine-tuning
1. Focus on best models
2. Try different hyperparameters
3. Train with more epochs (50-100)

### Day 4: Final Evaluation
1. Train best model with optimal settings
2. Evaluate thoroughly
3. Use `example_usage.py` for predictions

## 💡 Tips & Tricks

### Speed Up Training
- Use GPU: `GPU = 0`
- Increase batch size: `BATCH_SIZE = 512`
- Reduce epochs: `MAX_EPOCHS = 5`
- Fewer models: `MODELS = ['NHP', 'THP']`

### Improve Performance
- More epochs: `MAX_EPOCHS = 50`
- Larger model: `HIDDEN_SIZE = 128`
- Lower learning rate: `LEARNING_RATE = 5e-4`
- More layers: `num_layers = 4`

### Save Time
- Run overnight for comprehensive comparison
- Use CPU for small datasets (< 10K sequences)
- Use GPU for large datasets (> 50K sequences)

### Avoid Issues
- Start with small `MAX_EPOCHS` to test
- Check GPU availability before setting `GPU = 0`
- Ensure enough disk space for checkpoints
- Don't delete checkpoints until you're done

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | `cd .. && pip install -e .` |
| Out of memory | Reduce `BATCH_SIZE` or `HIDDEN_SIZE` |
| GPU error | Set `GPU = -1` to use CPU |
| Slow training | Increase `BATCH_SIZE` or use GPU |
| Poor results | Increase `MAX_EPOCHS` or tune hyperparameters |
| Notebook won't open | `jupyter notebook` in correct directory |

## 📈 Next Steps

After getting initial results:

1. **Hyperparameter Optimization**
   - Try different learning rates: [1e-4, 5e-4, 1e-3, 5e-3]
   - Try different hidden sizes: [32, 64, 128, 256]
   - Try different architectures: adjust `num_layers`, `num_heads`

2. **Extended Training**
   - Increase `MAX_EPOCHS` to 50-100
   - Monitor for overfitting (train vs. test NLL)
   - Use early stopping if available

3. **Model Analysis**
   - Use `example_usage.py` to load models
   - Analyze predictions on specific sequences
   - Identify failure cases

4. **Production Use**
   - Select best model
   - Retrain on full dataset (train + dev)
   - Deploy for real-world predictions

## 🎓 Learning Resources

- **EasyTPP Docs**: https://ant-research.github.io/EasyTemporalPointProcess/
- **GitHub**: https://github.com/ant-research/EasyTemporalPointProcess
- **Papers**: See model references in README.md

## ✅ Quick Checklist

Before running:
- [ ] Jupyter installed: `pip install jupyter`
- [ ] EasyTPP installed: `cd .. && pip install -e .`
- [ ] In correct directory: `cd tpp-demo`
- [ ] Data files present: `ls data/taxi/*.pkl`

First run:
- [ ] Open notebook: `jupyter notebook TPP_Model_Comparison.ipynb`
- [ ] Configure models and epochs
- [ ] Run all cells
- [ ] Check results in `results/` directory

After results:
- [ ] Review comparison table
- [ ] Check visualization plots
- [ ] Identify best model
- [ ] Note checkpoint location

---

**You're all set! Start with `QUICKSTART.md` and open the notebook. Good luck! 🚀**

