import torch
import numpy as np
from sklearn.datasets import make_blobs
import sys
sys.path.insert(0, ".")
from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
n_samples, n_features = X.shape
c = 3  # Number of clusters
m = 2  # Fuzziness parameter

# Initialize cluster centers and membership matrix
C = (torch.randn(c, n_features) * 0.001).requires_grad_(True)
W = torch.rand(n_samples, c)
W = W / W.sum(dim=1, keepdim=True)  # Normalize to ensure sum(W, axis=1) = 1

# Optimizer
optimizer = torch.optim.Adam([C, W], lr=1e1)

# Training loop
def train_gb(epochs=200):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute distances
        distances = torch.cdist(X, C)  # (n_samples, c)
        # Compute objective
        loss = torch.sum(W.pow(m) * distances.pow(2))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Normalize membership matrix
        W.data = W.data / W.data.sum(dim=1, keepdim=True)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

train_gb(epochs=10000)

print(C)

plot_assignment(X.detach().numpy(), C.detach().numpy(), W.detach().numpy())