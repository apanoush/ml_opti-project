# Exploring the Capabilities of Zeroth-Order Methods on ML tasks

This repository contains the implementation and experiments for the research project comparing zero-order optimization algorithms (SPSA vs. multi-point gradient estimators) applied to gradient descent across multiple problem domains. 

## Research Overview
Gradient estimation is crucial for optimization in black-box scenarios where explicit gradients are unavailable. This project empirically compares two prominent zero-order methods:
1. **Simultaneous Perturbation Stochastic Approximation (SPSA)**
2. **Multi-Point Gradient Estimators (MPGE)**
across four distinct problem domains:
- Linear Regression
- K-Means Clustering
- Reinforcement Learning
- Architecture Optimization

The study evaluates convergence speed, solution quality, and robustness under different noise conditions.

## Team
Anoush Azar-Pey - anoush.azar-pey@epfl.ch  
Clementine Jordan - clementine.jordan@epfl.ch  
Damien Genoud - damien.genoud@epfl.ch

## Repository Structure

- `eyperiments/`: All problems from the report
  - `algorithms.py`: Implementations of optimization methods
    - `spsa`: SPSA gradient estimator
    - `multi_point`: MPGE gradient estimators
    - `gradient_descent`: Gradient descent framework
  - `linear_regression/`
    - `linear_regression.py`: Script to run all 3 algorithms on a linear regression problem with chosen parameters
    - `plot.py` Script to plot all additional figures from the report along with computing the results tables
  - `soft_kmeans/`
    - `utils.py`: Contains key parameters for the data generation
    - `kmeans.py`: Script to run all 3 algorithms on a soft kmeans clustering problem with chosen parameters
    - `plot.py` Script to plot all additional figures from the report along with computing the results tables
  - `mountain_car/`
    - `mountain_car.py`: Script to run SPSA and MPGE on the Mountain Car task (Reinforcement Learning)
  - `NAS_spsa/`
    - `spsa_NAS.py`: Script to run SPSA and Grid-Search on a Neural Architecture Search task (NAS)

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