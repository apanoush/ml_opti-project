import numpy as np
from sklearn.datasets import make_blobs

N_CENTERS = [3, 5, 10][-1]

def plot_assignment(X, C_opt, W_opt):
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
    plt.show()

def generate_data(n_centers=N_CENTERS):
    # Generate synthetic data
    print(f"Creating {n_centers} blobs")
    X, y = make_blobs(n_samples=300, centers=n_centers, cluster_std=2., random_state=42, n_features=2)
    n_samples, n_features = X.shape
    c = n_centers  # Number of clusters
    m = 2  # Fuzziness parameter
    return X, y, n_samples, n_features, c , m