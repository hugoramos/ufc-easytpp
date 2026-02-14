"""
Temporal Point Process Model Comparison Script

This script trains and evaluates different TPP models on datasets and compares their NLL scores.
"""

import sys
import os
import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add the parent directory to path to import easy_tpp
sys.path.insert(0, os.path.abspath('..'))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# Results directory
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("TPP MODEL COMPARISON")
print("="*80)
print(f"Models to evaluate: {MODELS}")
print(f"Dataset: {DATASET_NAME}")
print(f"Results will be saved to: {RESULTS_DIR}")
print("="*80 + "\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_pkl_data(file_path):
    """Load and inspect pickle data file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_model_config(model_name, dataset_name, stage='train'):
    """Create configuration dictionary for a specific model."""
    
    config = {
        'pipeline_config_id': 'runner_config',
        'data': {
            dataset_name: {
                'data_format': 'pkl',
                'train_dir': f'./data/{dataset_name}/train.pkl',
                'valid_dir': f'./data/{dataset_name}/dev.pkl',
                'test_dir': f'./data/{dataset_name}/test.pkl',
                'data_specs': {
                    'num_event_types': 10,
                    'pad_token_id': 10,
                    'padding_side': 'right',
                    'truncation_side': 'right',
                    'max_len': 100
                }
            }
        },
        f'{model_name}_{stage}': {
            'base_config': {
                'stage': stage,
                'backend': 'torch',
                'dataset_id': dataset_name,
                'runner_id': 'std_tpp',
                'model_id': model_name,
                'base_dir': './checkpoints/'
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
                    'dtime_max': 5
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
    
    return config


def train_model(model_name, dataset_name):
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset_name} dataset")
    print(f"{'='*60}\n")
    
    try:
        # Create config
        config_dict = create_model_config(model_name, dataset_name, stage='train')
        
        # Save config to temporary file
        temp_config_path = f'./configs/temp_{model_name.lower()}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Build config and runner
        experiment_id = f'{model_name}_train'
        cfg = Config.build_from_yaml_file(temp_config_path, experiment_id=experiment_id)
        runner = Runner.build_from_config(cfg)
        
        # Run training
        runner.run()
        
        # Get checkpoint directory
        checkpoint_dir = runner.config.base_config.specs['base_dir']
        
        # Clean up temp config
        os.remove(temp_config_path)
        
        return {
            'model': model_name,
            'status': 'success',
            'checkpoint_dir': checkpoint_dir
        }
        
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }


def evaluate_model(model_name, dataset_name, checkpoint_dir):
    """Evaluate a trained model and return metrics."""
    print(f"\nEvaluating {model_name}...")
    
    try:
        # Create eval config
        config_dict = create_model_config(model_name, dataset_name, stage='eval')
        
        # Update with checkpoint directory
        experiment_id = f'{model_name}_eval'
        config_dict[experiment_id]['base_config']['pretrained_model_dir'] = f'{checkpoint_dir}/models/saved_model'
        
        # Save config to temporary file
        temp_config_path = f'./configs/temp_{model_name.lower()}_eval.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Build config and runner
        cfg = Config.build_from_yaml_file(temp_config_path, experiment_id=experiment_id)
        runner = Runner.build_from_config(cfg)
        
        # Run evaluation
        runner.run()
        
        # Try to extract metrics from output
        eval_output_path = f'{checkpoint_dir}/{model_name}_eval_output.yaml'
        
        metrics = {}
        if os.path.exists(eval_output_path):
            with open(eval_output_path, 'r') as f:
                eval_results = yaml.safe_load(f)
                metrics = eval_results.get('metrics', {})
        
        # Clean up temp config
        os.remove(temp_config_path)
        
        return {
            'model': model_name,
            'status': 'success',
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }


def extract_nll_from_logs(checkpoint_dir, model_name):
    """Extract NLL scores from training/evaluation logs."""
    nll_scores = {
        'train_nll': None,
        'valid_nll': None,
        'test_nll': None
    }
    
    # Try to read from output yaml files
    train_output = f'{checkpoint_dir}/{model_name}_train_output.yaml'
    eval_output = f'{checkpoint_dir}/{model_name}_eval_output.yaml'
    
    try:
        if os.path.exists(train_output):
            with open(train_output, 'r') as f:
                data = yaml.safe_load(f)
                if 'train_nll' in data:
                    nll_scores['train_nll'] = data['train_nll']
                if 'valid_nll' in data:
                    nll_scores['valid_nll'] = data['valid_nll']
        
        if os.path.exists(eval_output):
            with open(eval_output, 'r') as f:
                data = yaml.safe_load(f)
                if 'test_nll' in data:
                    nll_scores['test_nll'] = data['test_nll']
    except Exception as e:
        print(f"Error reading logs for {model_name}: {e}")
    
    return nll_scores


def plot_results(results_df, timestamp):
    """Create visualizations to compare model performance."""
    # Filter out None values for plotting
    plot_df = results_df.copy()
    plot_df = plot_df[plot_df['Test NLL'].notna()]
    
    if len(plot_df) == 0:
        print("No valid results to plot.")
        return
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Train NLL
    if plot_df['Train NLL'].notna().any():
        axes[0].bar(plot_df['Model'], plot_df['Train NLL'], color='steelblue')
        axes[0].set_title('Training NLL (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Negative Log-Likelihood')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
    
    # Valid NLL
    if plot_df['Valid NLL'].notna().any():
        axes[1].bar(plot_df['Model'], plot_df['Valid NLL'], color='coral')
        axes[1].set_title('Validation NLL (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Negative Log-Likelihood')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
    
    # Test NLL
    if plot_df['Test NLL'].notna().any():
        axes[2].bar(plot_df['Model'], plot_df['Test NLL'], color='seagreen')
        axes[2].set_title('Test NLL (Lower is Better)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Negative Log-Likelihood')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f'{RESULTS_DIR}/model_comparison_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # 1. Inspect dataset
    print("\n" + "="*80)
    print("DATASET INSPECTION")
    print("="*80 + "\n")
    
    train_data_dict = load_pkl_data(f'{DATA_DIR}/train.pkl')
    dev_data_dict = load_pkl_data(f'{DATA_DIR}/dev.pkl')
    test_data_dict = load_pkl_data(f'{DATA_DIR}/test.pkl')
    
    # Extract sequences from dict structure: {'dim_process': 10, 'train': [...]}
    train_data = train_data_dict.get('train', train_data_dict)
    dev_data = dev_data_dict.get('dev', dev_data_dict)
    test_data = test_data_dict.get('test', test_data_dict)
    
    print(f"Train sequences: {len(train_data)}")
    print(f"Dev sequences: {len(dev_data)}")
    print(f"Test sequences: {len(test_data)}")
    print(f"Dim process: {train_data_dict.get('dim_process', 'N/A')}")
    
    # 2. Train all models
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    training_results = []
    for model in MODELS:
        result = train_model(model, DATASET_NAME)
        training_results.append(result)
        print(f"\n{model}: {result['status']}")
    
    # 3. Evaluate all models
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    evaluation_results = []
    for result in training_results:
        if result['status'] == 'success':
            eval_result = evaluate_model(
                result['model'], 
                DATASET_NAME, 
                result['checkpoint_dir']
            )
            evaluation_results.append(eval_result)
    
    # 4. Compile results
    print("\n" + "="*80)
    print("RESULTS COMPILATION")
    print("="*80 + "\n")
    
    results_data = []
    for train_result in training_results:
        if train_result['status'] == 'success':
            model_name = train_result['model']
            checkpoint_dir = train_result['checkpoint_dir']
            
            nll_scores = extract_nll_from_logs(checkpoint_dir, model_name)
            
            results_data.append({
                'Model': model_name,
                'Train NLL': nll_scores['train_nll'],
                'Valid NLL': nll_scores['valid_nll'],
                'Test NLL': nll_scores['test_nll'],
                'Checkpoint': checkpoint_dir
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80 + "\n")
    print(results_df.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{RESULTS_DIR}/model_comparison_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # 5. Visualize results
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    plot_results(results_df, timestamp)
    
    # 6. Rank models
    if 'Test NLL' in results_df.columns:
        ranked_df = results_df.dropna(subset=['Test NLL']).sort_values('Test NLL')
        
        print("\n" + "="*80)
        print("MODEL RANKING (by Test NLL - Lower is Better)")
        print("="*80 + "\n")
        
        for idx, row in ranked_df.iterrows():
            rank = list(ranked_df.index).index(idx) + 1
            print(f"{rank}. {row['Model']:15s} - Test NLL: {row['Test NLL']:.4f}")
        
        # Best model
        if len(ranked_df) > 0:
            best_model = ranked_df.iloc[0]
            print(f"\n🏆 Best Model: {best_model['Model']} (Test NLL: {best_model['Test NLL']:.4f})")
            print(f"   Checkpoint: {best_model['Checkpoint']}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

