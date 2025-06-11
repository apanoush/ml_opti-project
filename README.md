# Comparative Analysis of Zero-Order Gradient Estimation Methods

This repository contains the implementation and experiments for the research project comparing zero-order optimization algorithms (SPSA vs. multi-point gradient estimators) applied to gradient descent across multiple problem domains. 

## Research Overview
Gradient estimation is crucial for optimization in black-box scenarios where explicit gradients are unavailable. This project empirically compares two prominent zero-order methods:
1. **Simultaneous Perturbation Stochastic Approximation (SPSA)**
2. **Multi-point gradient estimators**
across three distinct problem domains:
- Linear Regression
- K-Means Clustering
- Reinforcement Learning (Policy Optimization)

The study evaluates convergence speed, solution quality, and robustness under different noise conditions.

## Team
[Your Name] - [your.email@domain.edu]  
[Teammate 1 Name] - [teammate1.email@domain.edu]  
[Teammate 2 Name] - [teammate2.email@domain.edu]

## Repository Structure

### Core Components
- `algorithms.py`: Implementations of optimization methods
  - `spsa`: SPSA gradient estimator
  - `multi_point`: Multi-point gradient estimators
  - `gradient_descent`: Gradient descent framework
- `problems/`: Benchmark problems
  - `linear_regression/`
  - `kmeans/`
  - `reinforcement_learning/`
- `experiments/`: Experiment pipelines
  - `convergence_analysis/`
  - `noise_robustness/`
  - `hyperparameter_sensitivity/`
- `results/`: Output files
  - `figures/` (Generated plots)
  - `metrics/` (Quantitative results)
- `tests/`: Validation tests
- `utils/`: Helper functions

### Key Files
- `experiment_runner.py`: Main script to execute experiments
- `results_analysis.ipynb`: Jupyter notebook for results visualization
- `environment.yml`: Conda environment specification
- `requirements.txt`: Python dependencies

## Reproducibility

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/your-username/zero-order-comparison.git

# Create conda environment (recommended)
conda env create -f environment.yml

# Alternatively install via pip
pip install -r requirements.txt
```