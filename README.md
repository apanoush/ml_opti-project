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
Anoush Azar-Pey - anoush.azar-pey@epfl.ch  
Clementine Jordan - clementine.jordan@epfl.ch  
Damien Genoud - damien.genoud@epfl.ch

## Repository Structure

- `eyperiments/`: All problems from the report
  - `algorithms.py`: Implementations of optimization methods
    - `spsa`: SPSA gradient estimator
    - `multi_point`: Multi-point gradient estimators
    - `gradient_descent`: Gradient descent framework
  - `linear_regression/`
  - `soft_kmeans/`
  - `mountain_car/`
  - `NAS_spsa/`

### Key Files
- `report.pdf` Report of the project
- `experiments/results.ipynb`: Jupyter notebook for results visualization
- `requirements.txt`: Python dependencies

## Reproducibility

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/apanoush/ml_opti-project.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```