import torch
import numpy as np
from sklearn.datasets import make_blobs
import sys
sys.path.insert(0, ".")
from GB_ZO.kmeans_soft_clustering.kmeans import plot_assignment, generate_data
from GB_ZO.utils import compute_loss_and_plot
from tqdm import tqdm

epochs = 5000

X, y, n_samples, n_features, c , m = generate_data()

X = torch.tensor(X, dtype=torch.float32)

# Initialize cluster centers (better initialization)
C = torch.randn(c, n_features).requires_grad_(True)
c_history = np.zeros((epochs, *C.shape))
# Optimizer - only optimize cluster centers
optimizer = torch.optim.Adam([C], lr=1e-2)

def update_membership_matrix(X, C, m):
    """Update membership matrix using fuzzy c-means formula"""
    distances = torch.cdist(X, C)  # (n_samples, c)
    
    # Avoid division by zero
    distances = torch.clamp(distances, min=1e-8)
    
    # Compute membership matrix
    power = 2.0 / (m - 1)
    W = torch.zeros(n_samples, c)
    
    for i in range(c):
        denominator = torch.sum((distances[:, i:i+1] / distances) ** power, dim=1)
        W[:, i] = 1.0 / denominator
    
    return W

# Training loop
def train_gb(epochs=200):
    for i, epoch in tqdm(enumerate(range(epochs)), desc="optimization", unit="epoch", total=epochs):
        # Update membership matrix
        W = update_membership_matrix(X, C, m)
        
        optimizer.zero_grad()
        
        # Compute distances
        distances = torch.cdist(X, C)  # (n_samples, c)
        
        # Compute objective
        loss = torch.sum(W.pow(m) * distances.pow(2))
        
        # Backward pass
        loss.backward()
        c_history[i] = C.detach().numpy()
        optimizer.step()
        
        # if (epoch + 1) % 50 == 0:
        #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return W

W = train_gb(epochs=epochs)

def loss_function(C_np):
    # Ensure inputs are numpy arrays
    X_np = X.detach().numpy() if isinstance(X, torch.Tensor) else X
    W_np = W.detach().numpy() if isinstance(W, torch.Tensor) else W
    
    # Compute distances
    # distances_np = np.sqrt(np.sum((X_np[:, np.newaxis, :] - C_np[np.newaxis, :, :]) ** 2, axis=2)) # (n_samples, c)
    # A more direct way to compute squared Euclidean distances, which is what's needed before multiplying by W.pow(m)
    distances_sq_np = np.sum((X_np[:, np.newaxis, :] - C_np[np.newaxis, :, :]) ** 2, axis=2) # (n_samples, c)
    
    # Compute objective
    return np.sum((W_np ** m) * distances_sq_np)


print("Final cluster centers:")
print(C)

plot_assignment(X.detach().numpy(), C.detach().numpy(), W.detach().numpy())
compute_loss_and_plot(c_history, loss_function)