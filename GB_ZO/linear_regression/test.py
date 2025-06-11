import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from GB_ZO.algorithms import *

# ======== CONFIGURATION ========
np.random.seed(42)
MAX_ITERATIONS = 500
LR = 0.1
METHOD = ["analytical", "multi-point", "spsa"][1]  # Options: "analytical", "multi-point", "spsa"
PERTURB_SIZE = 0.1     # Used for ZO/SPSA methods
N_FEATURES = 10       # Number of input features (can be increased)
N_SAMPLES = 1000
TRUE_WEIGHTS = np.array([2.0] * N_FEATURES)  # Must match N_FEATURES
TRUE_BIAS = 3.0
# ===============================

# Generate synthetic data
def generate_data():
    X = np.random.rand(N_SAMPLES, N_FEATURES)
    y = X @ TRUE_WEIGHTS + TRUE_BIAS + np.random.normal(0, 0.4, N_SAMPLES)
    return X, y

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

# Main optimization function
def main():
    X, y = generate_data()
    initial_theta = init_theta()
    true_params = np.concatenate([TRUE_WEIGHTS, [TRUE_BIAS]])
    
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
    theta_opt, theta_history = gradient_descent(
        initial_theta,
        learning_rate=LR,
        max_iterations=MAX_ITERATIONS,
        gradient_function=grad_func,
        tolerance=1e-8
    )
    
    # Compute loss history
    loss_history = [mse_loss(theta, X, y) for theta in theta_history]
    param_error = [np.linalg.norm(theta - true_params) for theta in theta_history]
    
    # Print results
    print(f"\nFinal parameters ({METHOD}):")
    print(f"  Weights: {theta_opt[:-1]}")
    print(f"  Bias:    {theta_opt[-1]:.4f}")
    print(f"  Loss:    {loss_history[-1]:.6f}")
    print(f"  Param Error: {param_error[-1]:.6f}")
    
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

    # For 1D data: plot regression line
    if N_FEATURES == 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], y, alpha=0.6, c='gray', label='Data points')
        x_range = np.linspace(0, 1, 100)
        y_true = TRUE_WEIGHTS[0]*x_range + TRUE_BIAS
        y_pred = theta_opt[0]*x_range + theta_opt[1]
        plt.plot(x_range, y_true, 'g-', lw=3, label='True relationship')
        plt.plot(x_range, y_pred, 'r--', lw=2.5, label=f'Learned: $y={theta_opt[0]:.3f}x + {theta_opt[1]:.3f}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()