import numpy as np
import sys
sys.path.insert(0, ".")
from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment, generate_data
from GB_ZO.utils import compute_loss, plot_loss_history, output_result
from GB_ZO.algorithms import *

MAX_ITERATIONS = 10000
LR = [3e-3, 1e-5, 1e-3][-1]
METHOD = ["spsa", "multi-point", "analytical"][0]
ONLY_RESULTS = False

np.random.seed(42)

def analytical_gradient(theta):
    """Analytical gradient for fuzzy c-means cluster centers"""
    C = theta.reshape(c, n_features)
    W = compute_membership_matrix(X, C, m)
    U = W ** m  # Weight matrix (n_samples, c)
    
    # Compute gradient components
    weighted_sum = U.T @ X  # (c, n_features)
    weights_sum = np.sum(U, axis=0)  # (c,)
    grad_centers = -2 * (weighted_sum - weights_sum[:, np.newaxis] * C)
    
    return grad_centers.flatten()

FUNCTION = {
    "spsa": lambda x: spsa_gradient(
        x, objective, LR
    ),
    "multi-point": lambda x: multipoint_gradient_estimator(
        x, objective, LR
    ),
    "analytical": analytical_gradient
}[METHOD]

X, y, n_samples, n_features, c , m, sparse_dims = generate_data()

OUTPUT = f"GB_ZO/kmeans_soft_clustering/results/{METHOD}_{n_features}D.json"
OUTPUT_PLOT = f"GB_ZO/kmeans_soft_clustering/results/clusters2D.pdf" if METHOD == "spsa" else None
    

def main():
    initial_x = init_theta()

    print(f"Problem shape is {np.shape(initial_x)}")

    theta_opt, x_history = gradient_descent(
        initial_x, learning_rate=LR,
        max_iterations=MAX_ITERATIONS,
        gradient_function= FUNCTION,
        tolerance=1e-20
    )

    # Extract optimized parameters
    C_opt = theta_opt.reshape(c, n_features)
    W_opt = compute_membership_matrix(X, C_opt, m)

    print("Final cluster centers:")
    print(C_opt)
    print(f"Final objective value: {objective(theta_opt):.4f}")

    if n_features == 2:
        plot_assignment(X, C_opt, W_opt, output_path = OUTPUT_PLOT)

    loss_history = compute_loss(x_history, objective)
    if not ONLY_RESULTS: plot_loss_history(loss_history)
    additional_info = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": c,
        "method": METHOD,
        "lr": LR,
        "max_iterations": MAX_ITERATIONS,
        "sparse_dims": sparse_dims
    }
    output_result(theta_opt.tolist(), x_history.tolist(), loss_history.tolist(), OUTPUT, additional_info)


def compute_membership_matrix(X, C, m):
    """Compute membership matrix using fuzzy c-means formula"""
    n_samples, n_features = X.shape
    c = C.shape[0]
    
    # Compute distances
    distances = np.sqrt(((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-8)
    
    # Compute membership matrix
    power = 2.0 / (m - 1)
    W = np.zeros((n_samples, c))
    
    for i in range(c):
        denominator = np.sum((distances[:, i:i+1] / distances) ** power, axis=1)
        W[:, i] = 1.0 / denominator
    
    return W

# Initialize only cluster centers (not membership matrix)
def init_theta():
    # Initialize cluster centers near data points for better convergence
    indices = np.random.choice(n_samples, c, replace=False)
    C_init = X[indices] + np.random.normal(0, 0.1, (c, n_features))
    return C_init.flatten()

# Objective function for SPSA - only optimize cluster centers
def objective(theta):
    # Reshape to cluster centers
    C = np.reshape(theta, (c, n_features))
    
    # Compute membership matrix from current centers
    W = compute_membership_matrix(X, C, m)
    
    # Compute distances
    distances = np.sqrt(((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Compute fuzzy c-means objective
    loss = np.sum((W ** m) * (distances ** 2))
    
    return loss

# # Run SPSA optimization with better parameters
# theta_opt, x_history = spsa.minimize(
#     objective,
#     init_theta(),
#     iterations=500,
#     lr=0.1,           # Step size scaling 
#     px=0.1,           # Perturbation scaling
#     lr_decay=0.602,     # Step size decay
#     px_decay=0.101,     # Perturbation decay
# ) # A=50,            # Stability constant 

if __name__ == "__main__":
    main()