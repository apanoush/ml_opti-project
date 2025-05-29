import numpy as np
from sklearn.datasets import make_blobs
import sys
sys.path.insert(0, ".")
from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment
from GB_ZO.utils import compute_loss_and_plot
import GB_ZO.spsa as spsa

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
n_samples, n_features = X.shape
c = 3  # Number of clusters
m = 2  # Fuzziness parameter

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
    C = theta.reshape(c, n_features)
    
    # Compute membership matrix from current centers
    W = compute_membership_matrix(X, C, m)
    
    # Compute distances
    distances = np.sqrt(((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Compute fuzzy c-means objective
    loss = np.sum((W ** m) * (distances ** 2))
    
    return loss

# Run SPSA optimization with better parameters
theta_opt, x_history = spsa.minimize(
    objective,
    init_theta(),
    iterations=500,
    lr=0.1,           # Step size scaling 
    px=0.1,           # Perturbation scaling
    lr_decay=0.602,     # Step size decay
    px_decay=0.101,     # Perturbation decay
) # A=50,            # Stability constant 

# Extract optimized parameters
C_opt = theta_opt.reshape(c, n_features)
W_opt = compute_membership_matrix(X, C_opt, m)

print("Final cluster centers:")
print(C_opt)
print(f"Final objective value: {objective(theta_opt):.4f}")

plot_assignment(X, C_opt, W_opt)
compute_loss_and_plot(x_history, objective)