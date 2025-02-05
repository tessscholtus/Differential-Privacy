import torch
import torch.multiprocessing as mp
from itertools import product
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from dp_mnist import train_model as train_static
from dp_mnist_entropy import train_model as train_dynamic
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# Enable cuDNN auto-tuner
cudnn.benchmark = True

def load_mnist(batch_size=64, num_workers=4):
    """Load MNIST dataset with parallel data loading."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def evaluate_params(param_set, model_type, max_epsilon=1.0):
    """Evaluate a single parameter set with time tracking."""
    start_time = time.time()
    
    train_loader, test_loader = load_mnist()
    train_func = train_static if model_type == 'static' else train_dynamic
    
    try:
        metrics = train_func(train_loader, test_loader, param_set)
        
        # Check if privacy budget exceeded
        if metrics['epsilon_history'][-1] > max_epsilon:
            runtime = time.time() - start_time
            return {
                'params': param_set,
                'valid': False,
                'reason': f'Privacy budget exceeded: ε = {metrics["epsilon_history"][-1]:.2f}',
                'runtime': runtime
            }
        
        runtime = time.time() - start_time
        return {
            'params': param_set,
            'valid': True,
            'final_accuracy': metrics['accuracy_history'][-1],
            'final_epsilon': metrics['epsilon_history'][-1],
            'runtime': runtime,
            'train_loss': metrics['train_loss'][-1],
            'test_loss': metrics['test_loss'][-1]
        }
    except Exception as e:
        runtime = time.time() - start_time
        return {
            'params': param_set,
            'valid': False,
            'reason': str(e),
            'runtime': runtime
        }

def grid_search(param_grid, model_type='static', max_epsilon=1.0, n_processes=None):
    """Perform grid search with parallel processing."""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one core free
    
    # Generate all parameter combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    param_combinations = [dict(zip(param_keys, v)) for v in product(*param_values)]
    
    # Initialize multiprocessing pool
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=n_processes)
    
    # Run evaluations in parallel
    results = []
    for params in param_combinations:
        result = pool.apply_async(evaluate_params, (params, model_type, max_epsilon))
        results.append(result)
    
    # Close pool and collect results
    pool.close()
    pool.join()
    
    return [r.get() for r in results]

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameter grids for both models
    static_param_grid = {
        'learning_rate': [0.1, 0.15, 0.2],
        'max_grad_norm': [0.5, 1.0, 1.5],
        'base_noise_multiplier': [1.0, 1.1, 1.2],
        'epochs': [10, 15],
        'delta': [1e-5]
    }
    
    dynamic_param_grid = {
        **static_param_grid,
        'entropy_threshold': [0.3, 0.5, 0.7],
        'noise_scale_factor': [0.1, 0.2, 0.3]
    }
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'results/hypertuning/experiment_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run hyperparameter search for both models
    max_epsilon = 1.0  # Maximum privacy budget
    n_processes = mp.cpu_count() - 1  # Leave one core free
    
    print(f"Starting hyperparameter tuning with {n_processes} processes...")
    print(f"Maximum privacy budget (ε): {max_epsilon}")
    
    # Static model tuning
    print("\nTuning Static DP-SGD model...")
    static_results = grid_search(static_param_grid, 'static', max_epsilon, n_processes)
    
    # Dynamic model tuning
    print("\nTuning Dynamic DP-SGD model...")
    dynamic_results = grid_search(dynamic_param_grid, 'dynamic', max_epsilon, n_processes)
    
    # Save results
    results = {
        'static_model': {
            'param_grid': static_param_grid,
            'results': static_results
        },
        'dynamic_model': {
            'param_grid': dynamic_param_grid,
            'results': dynamic_results
        },
        'max_epsilon': max_epsilon,
        'n_processes': n_processes,
        'timestamp': timestamp
    }
    
    with open(results_dir / 'tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print best results
    def get_best_config(results_list):
        valid_results = [r for r in results_list if r['valid']]
        if not valid_results:
            return None
        return max(valid_results, key=lambda x: x['final_accuracy'])
    
    best_static = get_best_config(static_results)
    best_dynamic = get_best_config(dynamic_results)
    
    print("\nBest configurations:")
    if best_static:
        print("\nStatic DP-SGD:")
        print(f"Parameters: {best_static['params']}")
        print(f"Accuracy: {best_static['final_accuracy']:.4f}")
        print(f"Privacy budget (ε): {best_static['final_epsilon']:.4f}")
        print(f"Runtime: {best_static['runtime']:.2f} seconds")
    
    if best_dynamic:
        print("\nDynamic DP-SGD:")
        print(f"Parameters: {best_dynamic['params']}")
        print(f"Accuracy: {best_dynamic['final_accuracy']:.4f}")
        print(f"Privacy budget (ε): {best_dynamic['final_epsilon']:.4f}")
        print(f"Runtime: {best_dynamic['runtime']:.2f} seconds")
    
    print(f"\nFull results saved to: {results_dir / 'tuning_results.json'}")

if __name__ == "__main__":
    main()
