# 🚀 START HERE

## Welcome to TPP-DEMO!

This is your complete toolkit for comparing Temporal Point Process models.

---

## ⚡ Get Results in 3 Steps

### Step 1: Open the Notebook
```bash
jupyter notebook TPP_Model_Comparison.ipynb
```

### Step 2: Configure (Cell 4)
```python
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP']  # Which models?
MAX_EPOCHS = 10                            # How long to train?
GPU = -1                                   # CPU or GPU?
```

### Step 3: Run All Cells
Click: **Cell → Run All**

⏱️ Wait 30-60 minutes...

✅ Done! Check `results/` folder for:
- CSV table with NLL scores
- PNG plots comparing models

---

## 📚 Documentation Guide

| File | When to Read | Time |
|------|-------------|------|
| **START_HERE.md** (this file) | Right now! | 2 min |
| **QUICKSTART.md** | Before first run | 3 min |
| **OVERVIEW.md** | To understand everything | 10 min |
| **README.md** | For detailed reference | 15 min |

---

## 🎯 What You'll Get

### Comparison Table
```
Model    Test NLL
SAHP     2.215    ← Best!
THP      2.267
RMTPP    2.389
NHP      2.478
```

### Bar Charts
Visual comparison of all models

### Trained Models
Saved in `checkpoints/` for later use

---

## 💡 Quick Tips

**First time?** Start with:
```python
MODELS = ['NHP', 'RMTPP']  # Just 2 models
MAX_EPOCHS = 5              # Quick test
```

**Want better results?** Increase:
```python
MAX_EPOCHS = 50             # More training
HIDDEN_SIZE = 128           # Larger model
```

**Have a GPU?** Use it:
```python
GPU = 0                     # Much faster!
```

---

## 🆘 Problems?

| Issue | Fix |
|-------|-----|
| Can't import easy_tpp | `cd .. && pip install -e .` |
| Out of memory | Set `BATCH_SIZE = 64` |
| GPU error | Set `GPU = -1` |
| Notebook won't open | Run `jupyter notebook` in this directory |

---

## 📖 What to Read Next

1. ✅ You're reading **START_HERE.md** (you're here!)
2. 📖 Read **QUICKSTART.md** for detailed 3-step guide
3. 🎓 Read **OVERVIEW.md** to understand the full system
4. 📚 Read **README.md** when you need specific details

---

## 🎬 Ready?

```bash
# Open the notebook
jupyter notebook TPP_Model_Comparison.ipynb

# Or run the script
python model_comparison.py
```

**That's it! You're ready to go! 🚀**

---

## 🤔 Still Confused?

1. Open `QUICKSTART.md` - it has more details
2. Look at the notebook - it has explanations in each cell
3. Check `OVERVIEW.md` - it explains everything

**Don't overthink it - just open the notebook and run it!** ✨

