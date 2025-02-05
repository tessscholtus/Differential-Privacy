import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools
import json
from datetime import datetime
import os

@dataclass
class PrivacyMetrics:
    """Class to store privacy-related metrics during training"""
    epsilon_history: List[float]
    noise_levels: List[float]
    gradient_norms: List[float]
    accuracy_history: List[float]
    train_loss: List[float]
    test_loss: List[float]
    entropy_history: List[float] = None  # Optional, only for dynamic model

class PrivacyEvaluator:
    def __init__(self, delta: float = 1e-5, results_type: str = "baseline"):
        self.delta = delta
        self.static_metrics = None
        self.dynamic_metrics = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set results directory based on type
        if results_type == "baseline":
            self.results_dir = f"results/baseline/experiment_{self.timestamp}"
        else:
            self.results_dir = f"results/hypertuning/experiment_{self.timestamp}"
            
        os.makedirs(self.results_dir, exist_ok=True)
    
    def compute_privacy_loss_rate(self, metrics: PrivacyMetrics) -> float:
        """Compute the rate at which privacy is being consumed"""
        return np.mean(np.diff(metrics.epsilon_history))
    
    def compute_effective_noise(self, metrics: PrivacyMetrics) -> float:
        """Compute the effective noise level considering gradient norms"""
        return np.mean([n/g for n, g in zip(metrics.noise_levels, metrics.gradient_norms)])
    
    def save_metrics_to_file(self, metrics: Dict):
        """Save metrics to a JSON file"""
        metrics_file = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_file}")
    
    def plot_privacy_utility_tradeoff(self):
        """Plot privacy-utility trade-off curves for both methods"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy vs Privacy Loss
        plt.subplot(1, 2, 1)
        if self.static_metrics:
            plt.plot(self.static_metrics.epsilon_history, 
                    self.static_metrics.accuracy_history,
                    label='Static DP-SGD', marker='o')
            
        if self.dynamic_metrics:
            plt.plot(self.dynamic_metrics.epsilon_history,
                    self.dynamic_metrics.accuracy_history,
                    label='Dynamic DP-SGD', marker='x')
            
        plt.xlabel('Privacy Loss (ε)')
        plt.ylabel('Model Accuracy')
        plt.title('Privacy-Utility Trade-off')
        plt.legend()
        plt.grid(True)
        
        # Loss vs Privacy Loss
        plt.subplot(1, 2, 2)
        if self.static_metrics:
            plt.plot(self.static_metrics.epsilon_history, 
                    self.static_metrics.test_loss,
                    label='Static Test Loss', linestyle='--')
            plt.plot(self.static_metrics.epsilon_history, 
                    self.static_metrics.train_loss,
                    label='Static Train Loss')
            
        if self.dynamic_metrics:
            plt.plot(self.dynamic_metrics.epsilon_history,
                    self.dynamic_metrics.test_loss,
                    label='Dynamic Test Loss', linestyle='--')
            plt.plot(self.dynamic_metrics.epsilon_history,
                    self.dynamic_metrics.train_loss,
                    label='Dynamic Train Loss')
            
        plt.xlabel('Privacy Loss (ε)')
        plt.ylabel('Loss')
        plt.title('Loss vs Privacy Trade-off')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'privacy_utility_tradeoff.png'))
        plt.close()
    
    def plot_privacy_consumption(self):
        """Plot privacy budget consumption over time"""
        plt.figure(figsize=(15, 10))
        
        # Privacy loss over time
        plt.subplot(2, 2, 1)
        if self.static_metrics:
            plt.plot(self.static_metrics.epsilon_history, 
                    label='Static DP-SGD')
        if self.dynamic_metrics:
            plt.plot(self.dynamic_metrics.epsilon_history,
                    label='Dynamic DP-SGD')
        plt.xlabel('Training Steps')
        plt.ylabel('Privacy Loss (ε)')
        plt.title('Privacy Loss Over Time')
        plt.legend()
        plt.grid(True)
        
        # Noise levels
        plt.subplot(2, 2, 2)
        if self.static_metrics:
            plt.plot(self.static_metrics.noise_levels,
                    label='Static DP-SGD')
        if self.dynamic_metrics:
            plt.plot(self.dynamic_metrics.noise_levels,
                    label='Dynamic DP-SGD')
        plt.xlabel('Training Steps')
        plt.ylabel('Noise Multiplier')
        plt.title('Noise Levels Over Time')
        plt.legend()
        plt.grid(True)
        
        # Gradient norms
        plt.subplot(2, 2, 3)
        if self.static_metrics:
            plt.hist(self.static_metrics.gradient_norms, 
                    alpha=0.5, label='Static DP-SGD', bins=30)
        if self.dynamic_metrics:
            plt.hist(self.dynamic_metrics.gradient_norms,
                    alpha=0.5, label='Dynamic DP-SGD', bins=30)
        plt.xlabel('Gradient Norm')
        plt.ylabel('Frequency')
        plt.title('Gradient Norm Distribution')
        plt.legend()
        plt.grid(True)
        
        # Entropy (only for dynamic model)
        plt.subplot(2, 2, 4)
        if self.dynamic_metrics and self.dynamic_metrics.entropy_history:
            plt.plot(self.dynamic_metrics.entropy_history,
                    label='Model Entropy')
            plt.xlabel('Training Steps')
            plt.ylabel('Entropy')
            plt.title('Model Entropy Over Time')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Entropy tracking\nonly available for\ndynamic model',
                    ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'privacy_consumption.png'))
        plt.close()
    
    def save_experiment_summary(self, results: Dict):
        """Save a summary of the experiment results"""
        summary = {
            'timestamp': self.timestamp,
            'results': results,
            'parameters': {
                'delta': self.delta
            }
        }
        
        # Save summary to file
        summary_file = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved experiment summary to {summary_file}")
    
    def compare_methods(self) -> Dict:
        """Compare both methods and return key metrics"""
        results = {}
        
        if self.static_metrics and self.dynamic_metrics:
            results['final_epsilon'] = {
                'static': self.static_metrics.epsilon_history[-1],
                'dynamic': self.dynamic_metrics.epsilon_history[-1]
            }
            results['privacy_loss_rate'] = {
                'static': self.compute_privacy_loss_rate(self.static_metrics),
                'dynamic': self.compute_privacy_loss_rate(self.dynamic_metrics)
            }
            results['effective_noise'] = {
                'static': self.compute_effective_noise(self.static_metrics),
                'dynamic': self.compute_effective_noise(self.dynamic_metrics)
            }
            results['final_accuracy'] = {
                'static': self.static_metrics.accuracy_history[-1],
                'dynamic': self.dynamic_metrics.accuracy_history[-1]
            }
            results['final_loss'] = {
                'static': {
                    'train': self.static_metrics.train_loss[-1],
                    'test': self.static_metrics.test_loss[-1]
                },
                'dynamic': {
                    'train': self.dynamic_metrics.train_loss[-1],
                    'test': self.dynamic_metrics.test_loss[-1]
                }
            }
            
            # Save results
            self.save_experiment_summary(results)
        
        return results

def grid_search_with_privacy(model_class, train_func, hyperparameters: Dict, 
                           max_epsilon: float = 10.0) -> Tuple[Dict, PrivacyMetrics]:
    """Perform grid search while respecting privacy budget"""
    best_params = None
    best_metrics = None
    best_accuracy = 0
    
    for params in generate_param_combinations(hyperparameters):
        model = model_class(**params)
        metrics = train_func(model, params)
        
        # Check if we exceeded privacy budget
        if metrics.epsilon_history[-1] > max_epsilon:
            continue
            
        if metrics.accuracy_history[-1] > best_accuracy:
            best_accuracy = metrics.accuracy_history[-1]
            best_params = params
            best_metrics = metrics
    
    return best_params, best_metrics

def generate_param_combinations(hyperparameters: Dict):
    """Generate all possible hyperparameter combinations"""
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))
