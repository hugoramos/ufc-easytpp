# TPP-DEMO: Temporal Point Process Model Comparison

This directory contains tools for training and evaluating different Temporal Point Process (TPP) models and comparing their performance.

## 📁 Project Structure

```
tpp-demo/
├── TPP_Model_Comparison.ipynb  # Interactive Jupyter notebook for model comparison
├── model_comparison.py          # Standalone Python script (same functionality)
├── configs/                     # Model configuration files
│   └── taxi_nhp.yaml           # Example NHP configuration
├── data/                        # Dataset directory
│   └── taxi/                   # Taxi dataset
│       ├── train.pkl
│       ├── dev.pkl
│       └── test.pkl
├── checkpoints/                 # Trained model checkpoints
├── results/                     # Comparison results (CSV & plots)
└── scripts/                     # Utility scripts
```

## 🚀 Quick Start

### Option 1: Using Jupyter Notebook (Recommended)

1. **Launch Jupyter**:
   ```bash
   jupyter notebook TPP_Model_Comparison.ipynb
   ```

2. **Run the cells sequentially** to:
   - Configure models and datasets
   - Inspect dataset statistics
   - Train multiple TPP models
   - Evaluate and compare NLL scores
   - Visualize results

### Option 2: Using Python Script

```bash
python model_comparison.py
```

This will automatically:
- Train all configured models
- Evaluate them on the test set
- Generate comparison tables and plots
- Save results to the `results/` directory

## 🎯 Available Models

The following TPP models are available for comparison:

| Model | Description | Paper |
|-------|-------------|-------|
| **NHP** | Neural Hawkes Process | [NeurIPS'17](https://arxiv.org/abs/1612.09328) |
| **RMTPP** | Recurrent Marked TPP | [KDD'16](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) |
| **THP** | Transformer Hawkes Process | [ICML'20](https://arxiv.org/abs/2002.09291) |
| **SAHP** | Self-Attentive Hawkes Process | [ICML'20](https://arxiv.org/abs/1907.07561) |
| **FullyNN** | Fully Neural Network TPP | [NeurIPS'19](https://arxiv.org/abs/1905.09690) |
| **IntensityFree** | Intensity-Free TPP | [ICLR'20](https://arxiv.org/abs/1909.12127) |
| **ODETPP** | ODE-based TPP | [ICLR'21](https://arxiv.org/abs/2011.04583) |
| **AttNHP** | Attentive Neural Hawkes | [ICLR'22](https://arxiv.org/abs/2201.00044) |

## ⚙️ Configuration

### In the Notebook

Modify the configuration cell to customize your experiments:

```python
# Models to compare
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP']

# Dataset configuration
DATASET_NAME = 'taxi'
DATA_DIR = './data/taxi'

# Training configuration
BATCH_SIZE = 256
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
GPU = -1  # -1 for CPU, 0 for GPU
```

### In the Python Script

Edit the configuration section at the top of `model_comparison.py`.

## 📊 Output

### Results Files

After running the comparison, you'll find:

1. **CSV File**: `results/model_comparison_YYYYMMDD_HHMMSS.csv`
   - Contains NLL scores for all models
   - Includes train, validation, and test metrics

2. **Plot**: `results/model_comparison_YYYYMMDD_HHMMSS.png`
   - Bar charts comparing NLL across models
   - Separate plots for train/valid/test sets

### Checkpoints

Trained models are saved in `checkpoints/` with the following structure:

```
checkpoints/
└── <pid>_<timestamp>/
    ├── models/
    │   └── saved_model  # Trained model weights
    ├── log              # Training logs
    └── <MODEL>_train_output.yaml  # Training metrics
```

## 📈 Understanding NLL Scores

**Negative Log-Likelihood (NLL)** measures how well a model fits the data:

- **Lower NLL = Better performance**
- Compare across train/valid/test to detect overfitting
- NLL is the primary metric for TPP model evaluation

## 🔧 Advanced Usage

### Adding a New Model

1. Ensure the model is implemented in `easy_tpp`
2. Add the model name to the `MODELS` list
3. Run the notebook/script

### Using a Different Dataset

1. Place your dataset in `data/<dataset_name>/`
   - Required files: `train.pkl`, `dev.pkl`, `test.pkl`
2. Update `DATASET_NAME` and `DATA_DIR`
3. Adjust `num_event_types` in the config if needed

### Hyperparameter Tuning

Modify these parameters in the configuration:

- `HIDDEN_SIZE`: Model capacity (32, 64, 128, 256)
- `MAX_EPOCHS`: Training duration (10, 20, 50, 100)
- `LEARNING_RATE`: Optimization speed (1e-4, 1e-3, 1e-2)
- `BATCH_SIZE`: Memory vs. speed trade-off (64, 128, 256, 512)

### Evaluating Pre-trained Models

If you already have trained models in `checkpoints/`, you can skip training:

```python
# In the notebook, skip the training cell and directly run evaluation
checkpoint_dirs = {
    'NHP': './checkpoints/12345_timestamp/',
    'RMTPP': './checkpoints/67890_timestamp/',
}

evaluation_results = []
for model_name, checkpoint_dir in checkpoint_dirs.items():
    result = evaluate_model(model_name, DATASET_NAME, checkpoint_dir)
    evaluation_results.append(result)
```

## 📝 Example Workflow

1. **Explore the dataset**:
   ```python
   train_data = load_pkl_data('./data/taxi/train.pkl')
   print(f"Number of sequences: {len(train_data)}")
   ```

2. **Train models**:
   - Run the training cell in the notebook
   - Models are trained sequentially
   - Progress is displayed for each model

3. **Compare results**:
   - View the comparison table
   - Check the visualization plots
   - Identify the best-performing model

4. **Analyze**:
   - Look for overfitting (train vs. test NLL gap)
   - Consider computational cost vs. performance
   - Select the model that best fits your needs

## 🐛 Troubleshooting

### GPU Issues

If you encounter GPU errors, set `GPU = -1` to use CPU:

```python
GPU = -1  # Force CPU usage
```

### Memory Errors

Reduce batch size if you run out of memory:

```python
BATCH_SIZE = 64  # Smaller batch size
```

### Import Errors

Ensure EasyTPP is installed and accessible:

```bash
cd ..
pip install -e .
```

## 📚 References

- [EasyTPP Documentation](https://ant-research.github.io/EasyTemporalPointProcess/)
- [EasyTPP GitHub](https://github.com/ant-research/EasyTemporalPointProcess)

## 📄 License

This project follows the same license as EasyTemporalPointProcess.

