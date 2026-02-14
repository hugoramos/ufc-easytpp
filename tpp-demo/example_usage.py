"""
Example: How to use a trained TPP model for predictions

This script shows how to:
1. Load a trained model from checkpoints
2. Make predictions on new sequences
3. Evaluate model performance
"""

import sys
import os
import pickle
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# ============================================================================
# EXAMPLE 1: Load and Evaluate a Trained Model
# ============================================================================

def load_trained_model(checkpoint_dir, model_name='NHP'):
    """
    Load a trained model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., './checkpoints/12345_timestamp/')
        model_name: Name of the model (e.g., 'NHP', 'RMTPP', 'THP')
    
    Returns:
        runner: Loaded model runner
    """
    print(f"Loading {model_name} model from {checkpoint_dir}...")
    
    # Create evaluation config
    config = {
        'pipeline_config_id': 'runner_config',
        'data': {
            'taxi': {
                'data_format': 'pkl',
                'train_dir': './data/taxi/train.pkl',
                'valid_dir': './data/taxi/dev.pkl',
                'test_dir': './data/taxi/test.pkl',
                'data_specs': {
                    'num_event_types': 10,
                    'pad_token_id': 10,
                    'padding_side': 'right',
                    'truncation_side': 'right',
                    'max_len': 100
                }
            }
        },
        f'{model_name}_eval': {
            'base_config': {
                'stage': 'eval',
                'backend': 'torch',
                'dataset_id': 'taxi',
                'runner_id': 'std_tpp',
                'model_id': model_name,
                'base_dir': './checkpoints/',
                'pretrained_model_dir': f'{checkpoint_dir}/models/saved_model'
            },
            'model_config': {
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
                'batch_size': 256,
                'metrics': ['acc', 'rmse']
            }
        }
    }
    
    # Save temporary config
    temp_config_path = f'./configs/temp_load_{model_name.lower()}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Build runner
    cfg = Config.build_from_yaml_file(temp_config_path, experiment_id=f'{model_name}_eval')
    runner = Runner.build_from_config(cfg)
    
    # Clean up
    os.remove(temp_config_path)
    
    print(f"✓ Model loaded successfully!")
    return runner


# ============================================================================
# EXAMPLE 2: Evaluate on Test Set
# ============================================================================

def evaluate_on_test_set(checkpoint_dir, model_name='NHP'):
    """
    Evaluate a trained model on the test set.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        model_name: Name of the model
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on test set")
    print(f"{'='*60}\n")
    
    runner = load_trained_model(checkpoint_dir, model_name)
    
    # Run evaluation
    runner.run()
    
    # Try to load results
    eval_output_path = f'{checkpoint_dir}/{model_name}_eval_output.yaml'
    if os.path.exists(eval_output_path):
        with open(eval_output_path, 'r') as f:
            results = yaml.safe_load(f)
            print(f"\nEvaluation Results:")
            print(f"  Test NLL: {results.get('test_nll', 'N/A')}")
            print(f"  Metrics: {results.get('metrics', {})}")
            return results
    else:
        print("No evaluation output file found.")
        return None


# ============================================================================
# EXAMPLE 3: Inspect Model Predictions
# ============================================================================

def inspect_predictions(checkpoint_dir, model_name='NHP', num_samples=5):
    """
    Load test data and show model predictions.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        model_name: Name of the model
        num_samples: Number of samples to inspect
    """
    print(f"\n{'='*60}")
    print(f"Inspecting {model_name} predictions")
    print(f"{'='*60}\n")
    
    # Load test data
    with open('./data/taxi/test.pkl', 'rb') as f:
        test_data_dict = pickle.load(f)
    
    # Extract sequences from dict structure
    test_data = test_data_dict.get('test', test_data_dict)
    
    print(f"Test set size: {len(test_data)} sequences")
    print(f"Dim process: {test_data_dict.get('dim_process', 'N/A')}")
    print(f"\nShowing first {num_samples} sequences:\n")
    
    for i in range(min(num_samples, len(test_data))):
        seq = test_data[i]
        print(f"Sequence {i+1}:")
        
        if isinstance(seq, dict):
            for key, value in seq.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: length={len(value)}, sample={value[:3]}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {seq}")
        print()


# ============================================================================
# EXAMPLE 4: Compare Multiple Models
# ============================================================================

def compare_models(checkpoint_dirs):
    """
    Compare multiple trained models.
    
    Args:
        checkpoint_dirs: Dictionary mapping model names to checkpoint paths
                        e.g., {'NHP': './checkpoints/123/', 'RMTPP': './checkpoints/456/'}
    """
    print(f"\n{'='*60}")
    print(f"Comparing {len(checkpoint_dirs)} models")
    print(f"{'='*60}\n")
    
    results = []
    
    for model_name, checkpoint_dir in checkpoint_dirs.items():
        try:
            # Try to load existing results
            train_output = f'{checkpoint_dir}/{model_name}_train_output.yaml'
            eval_output = f'{checkpoint_dir}/{model_name}_eval_output.yaml'
            
            model_results = {'model': model_name}
            
            if os.path.exists(train_output):
                with open(train_output, 'r') as f:
                    data = yaml.safe_load(f)
                    model_results['train_nll'] = data.get('train_nll', 'N/A')
                    model_results['valid_nll'] = data.get('valid_nll', 'N/A')
            
            if os.path.exists(eval_output):
                with open(eval_output, 'r') as f:
                    data = yaml.safe_load(f)
                    model_results['test_nll'] = data.get('test_nll', 'N/A')
            
            results.append(model_results)
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    # Print comparison table
    print(f"\n{'Model':<15} {'Train NLL':<12} {'Valid NLL':<12} {'Test NLL':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['model']:<15} "
              f"{str(r.get('train_nll', 'N/A')):<12} "
              f"{str(r.get('valid_nll', 'N/A')):<12} "
              f"{str(r.get('test_nll', 'N/A')):<12}")
    
    # Find best model
    valid_results = [r for r in results if isinstance(r.get('test_nll'), (int, float))]
    if valid_results:
        best = min(valid_results, key=lambda x: x['test_nll'])
        print(f"\n🏆 Best Model: {best['model']} (Test NLL: {best['test_nll']:.4f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("TPP MODEL USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Inspect test data
    print("\n[Example 1] Inspecting test data...")
    inspect_predictions(None, num_samples=3)
    
    # Example 2: Load and evaluate a specific model
    # Uncomment and modify the checkpoint path:
    # checkpoint_dir = './checkpoints/61166_140704276434816_251110-174055'
    # evaluate_on_test_set(checkpoint_dir, model_name='NHP')
    
    # Example 3: Compare multiple models
    # Uncomment and modify the checkpoint paths:
    # checkpoint_dirs = {
    #     'NHP': './checkpoints/12345_timestamp/',
    #     'RMTPP': './checkpoints/67890_timestamp/',
    #     'THP': './checkpoints/11111_timestamp/',
    # }
    # compare_models(checkpoint_dirs)
    
    print("\n" + "="*60)
    print("To use these examples:")
    print("1. Train models using TPP_Model_Comparison.ipynb")
    print("2. Find checkpoint directories in ./checkpoints/")
    print("3. Uncomment and modify the examples above")
    print("="*60)

