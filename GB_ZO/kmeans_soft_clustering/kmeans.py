import numpy as np
from sklearn.datasets import make_blobs

N_FEATURES = [2, 6][0]
N_CENTERS = [3, 5, 10][1]
STD = [2., 5.][-1]
SPARSE_DIMS = [N_FEATURES // 2, 0][-1]

def plot_assignment(X, C_opt, W_opt, output_path = None):
    import matplotlib.pyplot as plt

    # Determine cluster assignments
    cluster_assignments = np.argmax(W_opt, axis=1)

    # Plot the data points and cluster centers
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', marker='o', label='Data points')
    plt.scatter(C_opt[:, 0], C_opt[:, 1], c='red', marker='x', s=100, label='Cluster centers')
    plt.title('Fuzzy C-Means Clustering with SPSA')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def generate_data(n_centers=N_CENTERS, sparse_dims=SPARSE_DIMS):
    # Generate synthetic data
    print(f"Creating {n_centers} blobs")
    X, y = make_blobs(n_samples=300, centers=n_centers, cluster_std=STD, random_state=42, n_features=N_FEATURES)
    
    # Make some dimensions sparse (all zeros) to increase optimization difficulty
    if sparse_dims > 0 and sparse_dims < N_FEATURES:
        np.random.seed(42)  # For reproducibility
        sparse_indices = np.random.choice(N_FEATURES, size=sparse_dims, replace=False)
        X[:, sparse_indices] = 0
        print(f"Made dimensions {sparse_indices} sparse (all zeros)")
    
    n_samples, n_features = X.shape
    c = n_centers  # Number of clusters
    m = 2  # Fuzziness parameter
    return X, y, n_samples, n_features, c , m, sparse_dims
