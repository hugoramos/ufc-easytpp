"""
StackOverflow TPP Model Comparison
Train and compare 4 TPP models on real StackOverflow data from SNAP.

Models: NHP, RMTPP, THP, SAHP
Dataset: StackOverflow temporal network (17.8M edges, 5K sequences)
"""

import sys
import os
import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import gzip
import shutil
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from easy_tpp.utils import RunnerPhase

# Configuration
MODELS = ['NHP', 'RMTPP', 'THP', 'SAHP']
BATCH_SIZE = 256
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
GPU = -1

NUM_SEQUENCES = 5000
MIN_EVENTS = 10
MAX_EVENTS = 200

RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global training history
training_history = {}

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def download_stackoverflow():
    """Download StackOverflow from SNAP."""
    print(f"\n{'='*60}")
    print("Downloading StackOverflow from SNAP Stanford")
    print(f"{'='*60}\n")
    
    raw_dir = './data/stackoverflow_raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    url = 'http://snap.stanford.edu/data/sx-stackoverflow-a2q.txt.gz'
    output_file = f'{raw_dir}/sx-stackoverflow-a2q.txt.gz'
    
    print(f"Downloading: {url}")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Downloaded")
        
        txt_file = output_file.replace('.gz', '')
        print(f"Extracting...")
        with gzip.open(output_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"✓ Extracted: {txt_file}")
        return txt_file
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def convert_to_easytpp(raw_file):
    """Convert SNAP format to EasyTPP."""
    if raw_file is None:
        return None
    
    print(f"\n{'='*60}")
    print("Converting to EasyTPP format")
    print(f"{'='*60}\n")
    
    user_events = defaultdict(list)
    
    with open(raw_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  Processed {i:,} edges...")
            
            parts = line.strip().split()
            if len(parts) == 3:
                src, dst, timestamp = parts
                user_events[src].append({
                    'dst': dst,
                    'timestamp': int(timestamp)
                })
    
    print(f"\n✓ Loaded {i+1:,} edges from {len(user_events):,} users")
    
    # Filter and sample
    filtered = {u: e for u, e in user_events.items() if MIN_EVENTS <= len(e) <= MAX_EVENTS}
    print(f"✓ Filtered to {len(filtered):,} users")
    
    sampled_users = list(filtered.keys())[:NUM_SEQUENCES]
    print(f"✓ Sampling {len(sampled_users)} sequences")
    
    # Convert
    sequences = []
    num_event_types = 10
    
    for user in sampled_users:
        events = sorted(filtered[user], key=lambda x: x['timestamp'])
        start_time = events[0]['timestamp']
        sequence = []
        prev_time = start_time
        
        for idx, event in enumerate(events, 1):
            sequence.append({
                'idx_event': idx,
                'type_event': int(event['dst']) % num_event_types,
                'time_since_start': (event['timestamp'] - start_time) / 86400.0,
                'time_since_last_event': (event['timestamp'] - prev_time) / 86400.0
            })
            prev_time = event['timestamp']
        
        sequences.append(sequence)
    
    # Split
    n = len(sequences)
    n_train = int(0.7 * n)
    n_dev = int(0.1 * n)
    
    train_data = sequences[:n_train]
    dev_data = sequences[n_train:n_train+n_dev]
    test_data = sequences[n_train+n_dev:]
    
    print(f"\n✓ Splits: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")
    
    # Save
    output_dir = './data/stackoverflow'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/train.pkl', 'wb') as f:
        pickle.dump({'train': train_data, 'dim_process': num_event_types}, f)
    with open(f'{output_dir}/dev.pkl', 'wb') as f:
        pickle.dump({'dev': dev_data, 'dim_process': num_event_types}, f)
    with open(f'{output_dir}/test.pkl', 'wb') as f:
        pickle.dump({'test': test_data, 'dim_process': num_event_types}, f)
    
    print(f"✓ Saved to: {output_dir}/")
    return num_event_types


def create_config(model_name, dim_process):
    """Create model configuration."""
    return {
        'pipeline_config_id': 'runner_config',
        'data': {
            'stackoverflow': {
                'data_format': 'pkl',
                'train_dir': './data/stackoverflow/train.pkl',
                'valid_dir': './data/stackoverflow/dev.pkl',
                'test_dir': './data/stackoverflow/test.pkl',
                'data_specs': {
                    'num_event_types': dim_process,
                    'pad_token_id': dim_process,
                    'padding_side': 'right',
                    'truncation_side': 'right',
                    'max_len': 200
                }
            }
        },
        f'{model_name}_train': {
            'base_config': {
                'stage': 'train',
                'backend': 'torch',
                'dataset_id': 'stackoverflow',
                'runner_id': 'std_tpp',
                'model_id': model_name,
                'base_dir': './checkpoints_stackoverflow/'
            },
            'model_config': {
                'hidden_size': HIDDEN_SIZE,
                'time_emb_size': 16,
                'num_layers': 2,
                'num_heads': 2,
                'dropout': 0.1,
                'use_ln': False,
                'thinning_params': {
                    'num_seq': 10,
                    'num_sample': 1,
                    'num_exp': 500,
                    'look_ahead_time': 10,
                    'patience_counter': 5,
                    'over_sample_rate': 5,
                    'num_samples_boundary': 5,
                    'dtime_max': 10
                }
            },
            'trainer_config': {
                'seed': 2019,
                'gpu': GPU,
                'batch_size': BATCH_SIZE,
                'max_epoch': MAX_EPOCHS,
                'optimizer': 'adam',
                'learning_rate': LEARNING_RATE,
                'valid_freq': 1,
                'use_tfb': False,
                'metrics': ['acc', 'rmse']
            }
        }
    }


def train_with_history(model_runner, model_name):
    """Train and track metrics."""
    train_loader = model_runner._data_loader.train_loader()
    valid_loader = model_runner._data_loader.valid_loader()
    test_loader = model_runner._data_loader.test_loader()
    max_epochs = model_runner.runner_config.trainer_config.max_epoch
    
    history = {
        'train_loglike': [],
        'valid_loglike': [],
        'test_loglike': [],
        'epochs': []
    }
    
    print(f"Training {model_name}...")
    
    for epoch in range(max_epochs):
        train_metrics = model_runner.run_one_epoch(train_loader, RunnerPhase.TRAIN)
        valid_metrics = model_runner.run_one_epoch(valid_loader, RunnerPhase.VALIDATE)
        test_metrics = model_runner.run_one_epoch(test_loader, RunnerPhase.VALIDATE)
        
        history['train_loglike'].append(train_metrics['loglike'])
        history['valid_loglike'].append(valid_metrics['loglike'])
        history['test_loglike'].append(test_metrics['loglike'])
        history['epochs'].append(epoch)
        
        updated = model_runner.metrics_tracker.update_best("loglike", valid_metrics['loglike'], epoch)
        if updated:
            model_runner.model_wrapper.save(model_runner.runner_config.base_config.specs['saved_model_dir'])
        
        if epoch % 2 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch:2d}: train={train_metrics['loglike']:.4f}, "
                  f"valid={valid_metrics['loglike']:.4f}, test={test_metrics['loglike']:.4f}")
    
    model_runner.model_wrapper.close_summary()
    training_history[model_name] = history
    
    return test_metrics


def train_all():
    """Train all models."""
    results = {}
    
    for model_name in MODELS:
        try:
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}\n")
            
            config_dict = create_config(model_name, 10)
            config_path = f'{RESULTS_DIR}/{model_name}_config.yaml'
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            config = Config.build_from_yaml_file(config_path, experiment_id=f'{model_name}_train')
            model_runner = Runner.build_from_config(config)
            
            test_metrics = train_with_history(model_runner, f'stackoverflow_{model_name}')
            
            print(f"\n{model_name} Results:\")\n",
            for metric, value in test_metrics.items():
                print(f\"  {metric}: {value:.4f}\")\n",
            \n",
            results[model_name] = test_metrics\n",
            print(f\"\\n✓ {model_name} complete!\")\n",
            \n",
        except Exception as e:
            print(f\"\\n✗ Error: {str(e)}\")\n",
            results[model_name] = None\n",
    \n",
    return results\n",
\n",
\n",
def plot_learning_curves():\n",
    \"\"\"Plot learning curves with separate scales.\"\"\"\n",
    if not training_history:\n",
        print(\"No training history.\")\n",
        return\n",
    \n",
    n = len(training_history)\n",
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))\n",
    \n",
    if n == 1:\n",
        axes = axes.reshape(1, -1)\n",
    \n",
    colors = sns.color_palette('husl', n)\n",
    \n",
    for row, (model_name, history) in enumerate(training_history.items()):\n",
        name = model_name.replace('stackoverflow_', '')\n",
        \n",
        # Convert to NLL (loss)\n",
        train_nll = [-x for x in history['train_loglike']]\n",
        valid_nll = [-x for x in history['valid_loglike']]\n",
        test_nll = [-x for x in history['test_loglike']]\n",
        \n",
        # Plot\n",
        for col, (data, title, marker) in enumerate([\n",
            (train_nll, 'Training', 'o'),\n",
            (valid_nll, 'Validation', 's'),\n",
            (test_nll, 'Test', '^')\n",
        ]):\n",
            ax = axes[row, col] if n > 1 else axes[col]\n",
            ax.plot(history['epochs'], data, linewidth=3, color=colors[row],\n",
                   marker=marker, markersize=6, alpha=0.8, label=name)\n",
            ax.set_xlabel('Epoch', fontweight='bold')\n",
            ax.set_ylabel('Loss (NLL)', fontweight='bold')\n",
            ax.set_title(f'{name} - {title} Loss ↓', fontweight='bold')\n",
            ax.grid(True, alpha=0.3, linestyle='--')\n",
            ax.legend()\n",
            \n",
            reduction = data[0] - data[-1]\n",
            ax.text(0.02, 0.98, f'↓ {reduction:.2f}', transform=ax.transAxes,\n",
                   fontsize=10, va='top', bbox=dict(boxstyle='round', alpha=0.5))\n",
    \n",
    plt.tight_layout()\n",
    plt.savefig(f'{RESULTS_DIR}/learning_curves_separate.png', dpi=150, bbox_inches='tight')\n",
    print(f\"Saved: {RESULTS_DIR}/learning_curves_separate.png\")\n",
    plt.show()\n",
    \n",
    # Summary\n",
    print(f\"\\n{'='*70}\")\n",
    print(\"Loss Reduction Summary\")\n",
    print(f\"{'='*70}\\n\")\n",
    \n",
    for model_name, history in training_history.items():\n",
        name = model_name.replace('stackoverflow_', '')\n",
        initial = -history['test_loglike'][0]\n",
        final = -history['test_loglike'][-1]\n",
        reduction = initial - final\n",
        pct = (reduction / initial) * 100 if initial != 0 else 0\n",
        print(f\"{name}:\")\n",
        print(f\"  Test Loss: {initial:.2f} → {final:.2f} (↓ {reduction:.2f}, {pct:.1f}%)\")\n",


def plot_normalized_curves():
    \"\"\"Plot normalized curves - all models together.\"\"\"\n",
    if not training_history:\n",
        print(\"No training history.\")\n",
        return\n",
    \n",
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    colors = sns.color_palette('husl', len(training_history))\n",
    \n",
    for idx, (model_name, history) in enumerate(training_history.items()):\n",
        name = model_name.replace('stackoverflow_', '')\n",
        \n",
        # Convert to NLL and normalize\n",
        train_nll = [-x for x in history['train_loglike']]\n",
        valid_nll = [-x for x in history['valid_loglike']]\n",
        test_nll = [-x for x in history['test_loglike']]\n",
        \n",
        train_norm = [(v / train_nll[0]) * 100 for v in train_nll]\n",
        valid_norm = [(v / valid_nll[0]) * 100 for v in valid_nll]\n",
        test_norm = [(v / test_nll[0]) * 100 for v in test_nll]\n",
        \n",
        # Plot\n",
        for ax, data, marker in zip(axes, [train_norm, valid_norm, test_norm], ['o', 's', '^']):\n",
            ax.plot(history['epochs'], data, linewidth=3, color=colors[idx],\n",
                   marker=marker, markersize=5, alpha=0.8, label=name)\n",
    \n",
    for ax, title in zip(axes, ['Training', 'Validation', 'Test']):\n",
        ax.set_xlabel('Epoch', fontweight='bold')\n",
        ax.set_ylabel('Loss (% of Initial)', fontweight='bold')\n",
        ax.set_title(f'{title} - Relative Loss ↓', fontweight='bold')\n",
        ax.set_ylim(-5, 105)\n",
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)\n",
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5)\n",
        ax.grid(True, alpha=0.3)\n",
        ax.legend()\n",
        ax.invert_yaxis()\n",
    \n",
    plt.tight_layout()\n",
    plt.savefig(f'{RESULTS_DIR}/learning_curves_normalized.png', dpi=150, bbox_inches='tight')\n",
    print(f\"Saved: {RESULTS_DIR}/learning_curves_normalized.png\")\n",
    plt.show()


def main():
    \"\"\"Main execution.\"\"\"\n",
    print(\"=\"*70)\n",
    print(\"TPP Model Comparison - StackOverflow Dataset\")\n",
    print(\"=\"*70)\n",
    print(f\"Models: {', '.join(MODELS)}\")\n",
    print(f\"Sequences: {NUM_SEQUENCES}\")\n",
    print(f\"Epochs: {MAX_EPOCHS}\")\n",
    print(f\"Batch size: {BATCH_SIZE}\\n\")\n",
    \n",
    # Step 1: Download\n",
    print(\"\\n📥 STEP 1: Download Data\")\n",
    file = download_stackoverflow()\n",
    \n",
    # Step 2: Convert\n",
    print(\"\\n🔄 STEP 2: Convert to EasyTPP Format\")\n",
    dim_process = convert_to_easytpp(file)\n",
    \n",
    if dim_process is None:\n",
        print(\"\\n❌ Failed to prepare data. Exiting.\")\n",
        return\n",
    \n",
    # Step 3: Train\n",
    print(\"\\n🏋️ STEP 3: Train Models\")\n",
    results = train_all()\n",
    \n",
    # Step 4: Results\n",
    print(\"\\n📊 STEP 4: Final Results\")\n",
    print(\"=\"*70)\n",
    \n",
    results_list = []\n",
    for model, metrics in results.items():\n",
        if metrics:\n",
            results_list.append({\n",
                'Model': model,\n",
                'Log-Likelihood': metrics.get('loglike', 'N/A')\n",
            })\n",
    \n",
    df = pd.DataFrame(results_list)\n",
    df = df.sort_values('Log-Likelihood', ascending=False)\n",
    print(df.to_string(index=False))\n",
    \n",
    if len(df) > 0:\n",
        print(f\"\\n🏆 BEST MODEL: {df.iloc[0]['Model']} ({df.iloc[0]['Log-Likelihood']:.4f})\")\n",
    \n",
    # Step 5: Visualize\n",
    print(\"\\n📈 STEP 5: Learning Curves\")\n",
    print(\"\\n1. Separate Scales (each model's own y-axis):\")\n",
    plot_learning_curves()\n",
    \n",
    print(\"\\n2. Normalized Comparison (all on same scale):\")\n",
    plot_normalized_curves()\n",
    \n",
    print(\"\\n✅ Complete!\")\n",
    print(f\"Results saved to: {RESULTS_DIR}/\")\n",


if __name__ == '__main__':
    main()

