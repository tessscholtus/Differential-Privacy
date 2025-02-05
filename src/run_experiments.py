import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from dp_mnist import DPCNN, train_model as train_static_model
from dp_mnist_entropy import DPCNNEntropy, train_model as train_dynamic_model
from privacy_evaluation import PrivacyEvaluator, PrivacyMetrics

def load_mnist_data(batch_size: int = 250):
    """Load and prepare MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_baseline_comparison():
    """Run baseline comparison between static and dynamic DP-SGD"""
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    print("\nInitializing privacy evaluator...")
    privacy_evaluator = PrivacyEvaluator(delta=1e-5, results_type="baseline")
    
    # Train static model
    print("\nTraining static DP-SGD model...")
    static_metrics = train_static_model(train_loader, test_loader)
    privacy_evaluator.static_metrics = PrivacyMetrics(**static_metrics)
    
    # Train dynamic model
    print("\nTraining dynamic DP-SGD model...")
    dynamic_metrics = train_dynamic_model(train_loader, test_loader)
    privacy_evaluator.dynamic_metrics = PrivacyMetrics(**dynamic_metrics)
    
    # Generate plots and save results
    print("\nGenerating comparison plots...")
    privacy_evaluator.plot_privacy_utility_tradeoff()
    privacy_evaluator.plot_privacy_consumption()
    
    print("\nComparing methods...")
    results = privacy_evaluator.compare_methods()
    
    # Print summary
    print("\nResults Summary:")
    print("Static DP-SGD:")
    print(f"Final Accuracy: {results['final_accuracy']['static']:.4f}")
    print(f"Final ε: {results['final_epsilon']['static']:.2f}")
    print(f"Privacy Loss Rate: {results['privacy_loss_rate']['static']:.4f}\n")
    
    print("Dynamic DP-SGD:")
    print(f"Final Accuracy: {results['final_accuracy']['dynamic']:.4f}")
    print(f"Final ε: {results['final_epsilon']['dynamic']:.2f}")
    print(f"Privacy Loss Rate: {results['privacy_loss_rate']['dynamic']:.4f}")

if __name__ == "__main__":
    run_baseline_comparison()
