# DP-SGD Methods Comparison

This project compares static and dynamic Differential Privacy (DP) methods using Rényi privacy accounting. The static method is based on Abadi's DP-SGD, while the dynamic method is based on Zhang's entropy-based approach.

## Project Structure

```
dp_thesis/
├── src/                    # Source code
│   ├── dp_mnist.py        # Static DP-SGD implementation
│   ├── dp_mnist_entropy.py # Dynamic DP-SGD implementation
│   ├── privacy_evaluation.py # Privacy metrics and evaluation
│   └── run_experiments.py  # Main experiment runner
├── results/
│   ├── baseline/          # Results without hyperparameter tuning
│   └── hypertuning/       # Results with hyperparameter tuning
├── requirements.txt       # Project dependencies
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run baseline experiments:
```bash
python src/run_experiments.py
```

## Results

### Baseline Results
The baseline results (without hyperparameter tuning) are stored in `results/baseline/`. Each experiment creates a timestamped directory containing:
- `privacy_utility_tradeoff.png`: Visualization of accuracy vs privacy loss
- `privacy_consumption.png`: Detailed privacy metrics over time
- `experiment_summary.json`: Numerical results and parameters

### Hyperparameter Tuning
Results from hyperparameter tuning experiments are stored in `results/hypertuning/` with similar structure to baseline results.
