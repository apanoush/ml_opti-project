import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from GB_ZO.algorithms import *
from GB_ZO.utils import *

# ======== CONFIGURATION ========
np.random.seed(42)
MAX_ITERATIONS = [8000, 1000][-1]
LR = 0.1
METHOD = ["analytical", "multi-point", "spsa"][-1]  # Options: "analytical", "multi-point", "spsa"
PERTURB_SIZE = LR #0.1     # Used for ZO/SPSA methods
N_FEATURES = [10, 1][-1]       # Number of input features (can be increased)
N_SAMPLES = 1000
TRUE_WEIGHTS = np.array([2.0] * N_FEATURES)  # Must match N_FEATURES
TRUE_BIAS = 3.0
SPARSE_DIMS = [N_FEATURES // 2, 0][0]
OUTPUT = f"GB_ZO/linear_regression/results/{METHOD}_{N_FEATURES}D.json"
OUTPUT_PLOT = f"GB_ZO/linear_regression/results/linear_regression2D.pdf" if METHOD == "spsa" else None
# ===============================


# Generate synthetic data
def generate_data():
    X = np.random.rand(N_SAMPLES, N_FEATURES)

    # Make some dimensions sparse (all zeros) to increase optimization difficulty
    if SPARSE_DIMS > 0 and SPARSE_DIMS < N_FEATURES:
        np.random.seed(42)  # For reproducibility
        sparse_indices = np.random.choice(N_FEATURES, size=SPARSE_DIMS, replace=False)
        X[:, sparse_indices] = 0

        # Also set the corresponding true weights to 0
        TRUE_WEIGHTS[sparse_indices] = 0
        
        print(f"Made dimensions {sparse_indices} sparse (all zeros)")
        print(f"Set corresponding true weights to 0: {TRUE_WEIGHTS}")

    y = X @ TRUE_WEIGHTS + TRUE_BIAS + np.random.normal(0, 0.4, N_SAMPLES)
    return X, y, TRUE_WEIGHTS

# Mean Squared Error loss
def mse_loss(theta, X, y):
    w = theta[:-1]
    b = theta[-1]
    y_pred = X @ w + b
    return np.mean((y_pred - y)**2)

# Analytical gradient (for classical GD)
def analytical_gradient(theta, X, y):
    w = theta[:-1]
    b = theta[-1]
    y_pred = X @ w + b
    error = y_pred - y
    dw = 2 * X.T @ error / len(y)
    db = 2 * np.mean(error)
    return np.concatenate([dw, [db]])

# Initialize parameters
def init_theta():
    return np.zeros(N_FEATURES + 1)  # Weights + bias

def loss_plot(loss_history, param_error):
     # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title(f'Loss Convergence ({METHOD})')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(param_error)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Error (L2 Norm)')
    plt.title('Distance to True Parameters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_2d_results(X, y, final_x):
    plt.figure(figsize=(9, 3)) #(10, 6)
    plt.scatter(X[:, 0], y, alpha=0.3, c='gray', label='Data points')
    x_range = np.linspace(0, 1, 100)
    y_true = TRUE_WEIGHTS[0]*x_range + TRUE_BIAS
    y_pred = final_x[0]*x_range + final_x[1]
    plt.plot(x_range, y_true, 'g-', lw=3, label='True relationship: $y=2x + 3$')
    plt.plot(x_range, y_pred, 'r--', lw=2.5, label=f'Learned: $y={final_x[0]:.3f}x + {final_x[1]:.3f}$')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.show()

# Main optimization function
def main():
    X, y, TRUE_WEIGHTS = generate_data()
    initial_theta = init_theta()
    true_params = np.concatenate([TRUE_WEIGHTS, [TRUE_BIAS]])

    print(true_params)
    
    # Configure gradient function based on method
    if METHOD == "analytical":
        grad_func = lambda theta: analytical_gradient(theta, X, y)
    elif METHOD == "multi-point":
        grad_func = lambda theta: multipoint_gradient_estimator(
            theta, lambda t: mse_loss(t, X, y), K=PERTURB_SIZE
        )
    elif METHOD == "spsa":
        grad_func = lambda theta: spsa_gradient(
            theta, lambda t: mse_loss(t, X, y), K=PERTURB_SIZE
        )
    
    # Run optimization
    final_x, x_history = gradient_descent(
        initial_theta,
        learning_rate=LR,
        max_iterations=MAX_ITERATIONS,
        gradient_function=grad_func,
        tolerance=None #1e-15
    )
    
    # Compute loss history
    loss_history = [mse_loss(theta, X, y) for theta in x_history]
    param_error = [np.linalg.norm(theta - true_params) for theta in x_history]
    
    # Print results
    print(f"\nFinal parameters ({METHOD}):")
    print(f"  Weights: {final_x[:-1]}")
    print(f"  Bias:    {final_x[-1]:.4f}")
    print(f"  Loss:    {loss_history[-1]:.6f}")
    print(f"  Param Error: {param_error[-1]:.6f}")

    additional_info = {
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "method": METHOD,
        "lr": LR,
        "max_iterations": MAX_ITERATIONS,
        "sparse_dims": SPARSE_DIMS,
        "param_error": param_error
    }
    output_result(final_x.tolist(), x_history.tolist(), loss_history, OUTPUT, additional_info)
    
    loss_plot(loss_history, param_error)

    # For 1D data: plot regression line
    if N_FEATURES == 1:
        plot_2d_results(X, y, final_x)
        

if __name__ == "__main__":
    main()