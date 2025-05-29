import numpy as np
import spsa 
from sklearn.datasets import make_blobs
import sys
sys.path.insert(0, ".")
from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment
#from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment
# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
n_samples, n_features = X.shape
c = 3  # Number of clusters
m = 2  # Fuzziness parameter



# Flatten cluster centers and membership matrix into a single vector
def init_theta():
    C_flat = np.random.randn(c * n_features)  # 1D array for cluster centers
    W_flat = np.random.rand(n_samples * c)    # 1D array for membership weights
    
    # Normalize membership weights
    W = W_flat.reshape(n_samples, c)          # Reshape to 2D for normalization
    W = W / W.sum(axis=1, keepdims=True)      # Normalize rows to sum to 1
    W_flat = W.flatten()                      # Flatten back to 1D for concatenation
    
    return np.concatenate([C_flat, W_flat])   # Now both arrays are 1D
# Objective function for SPSA
def objective(theta):
    C_flat = theta[:c * n_features]
    W_flat = theta[c * n_features:]
    
    # Reshape
    C = C_flat.reshape(c, n_features)
    W = W_flat.reshape(n_samples, c)
    W = W / W.sum(axis=1, keepdims=True)  # Ensure valid membership matrix
    
    # Compute distances
    distances = np.array([np.linalg.norm(x - c_j) for x in X for c_j in C]).reshape(n_samples, c)
    # Compute objective
    loss = np.sum((W ** m) * (distances ** 2))
    return loss

# Run SPSA optimization
theta_opt, x_history = spsa.minimize(
    objective,
    init_theta(),
    
    iterations=200,
)

# a=0.16,
#     A=100,
#     c=0.26,
#     alpha=0.602,
#     gamma=0.101,

# Extract optimized parameters
C_opt = theta_opt[:c * n_features].reshape(c, n_features)
W_opt = theta_opt[c * n_features:].reshape(n_samples, c)
W_opt = W_opt / W_opt.sum(axis=1, keepdims=True)  # Final normalization

plot_assignment(X, C_opt, W_opt)