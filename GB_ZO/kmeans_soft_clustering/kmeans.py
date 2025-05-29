import numpy as np

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