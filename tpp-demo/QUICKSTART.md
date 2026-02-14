# Quick Start Guide

## 🎯 Goal

Compare different TPP models (NHP, RMTPP, THP, SAHP, etc.) on your dataset and see which one has the best NLL (Negative Log-Likelihood) score.

## 🚀 3 Steps to Get Results

### Step 1: Open the Notebook

```bash
cd /Users/hugoramossoares/Sites/EasyTemporalPointProcess/tpp-demo
jupyter notebook TPP_Model_Comparison.ipynb
```

### Step 2: Configure Your Experiment

In the **Configuration** cell, set:

```python
# Which models do you want to compare?
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP']

# How many epochs? (10 is quick, 50+ for better results)
MAX_EPOCHS = 10

# CPU or GPU? (-1 for CPU, 0 for GPU)
GPU = -1
```

### Step 3: Run All Cells

Click **Cell → Run All** or press **Shift+Enter** through each cell.

## 📊 What You'll Get

After ~10-30 minutes (depending on dataset size and number of models):

1. **Comparison Table**:
   ```
   Model    Train NLL  Valid NLL  Test NLL
   NHP      2.345      2.456      2.478
   RMTPP    2.234      2.367      2.389
   THP      2.123      2.245      2.267
   SAHP     2.089      2.198      2.215  ← Best!
   ```

2. **Bar Charts**: Visual comparison of NLL scores

3. **CSV File**: `results/model_comparison_YYYYMMDD_HHMMSS.csv`

4. **Trained Models**: Saved in `checkpoints/` for later use

## 🏆 Interpreting Results

- **Lower NLL = Better model**
- The model with the **lowest Test NLL** is the winner
- Check if Train NLL << Test NLL → possible overfitting

## ⚡ Quick Tips

### Run Faster
```python
MAX_EPOCHS = 5          # Fewer epochs
BATCH_SIZE = 512        # Larger batches
MODELS = ['NHP', 'THP'] # Fewer models
```

### Run Better
```python
MAX_EPOCHS = 50         # More epochs
LEARNING_RATE = 5e-4    # Fine-tune LR
HIDDEN_SIZE = 128       # Larger model
```

### Use GPU
```python
GPU = 0  # Use first GPU (if available)
```

## 🔄 Re-run with Different Settings

Want to try different hyperparameters? Just:

1. Change the configuration cell
2. Run all cells again
3. Compare the new results with previous ones

All results are timestamped, so nothing gets overwritten!

## 📞 Need Help?

- **Notebook not working?** Make sure you're in the right directory
- **Import errors?** Install EasyTPP: `cd .. && pip install -e .`
- **Out of memory?** Reduce `BATCH_SIZE` or `HIDDEN_SIZE`
- **Want more models?** Add to `MODELS` list: `['NHP', 'RMTPP', 'THP', 'SAHP', 'FullyNN', 'ODETPP']`

## 🎓 Next Steps

After finding the best model:

1. **Train longer**: Increase `MAX_EPOCHS` to 50-100
2. **Tune hyperparameters**: Adjust `LEARNING_RATE`, `HIDDEN_SIZE`
3. **Use the model**: Load from `checkpoints/` for predictions
4. **Try other datasets**: Change `DATASET_NAME` and `DATA_DIR`

---

**Happy modeling! 🚀**

