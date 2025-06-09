import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
sys.path.insert(0, ".")
from GB_ZO.utils import compute_loss_and_plot
import GB_ZO.spsa as spsa
from GB_ZO.algorithms import *

num_iterations = 2000

# Load and preprocess MNIST dataset with PyTorch
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert to NumPy arrays and flatten images
X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0  # Normalize and flatten
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

# Neural Network Architecture (parameters stored in a flat vector)
input_size = 784
hidden_size = 128
output_size = 10
theta_size = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

# Forward pass function
def forward(X, theta):
    idx = 0
    W1 = theta[idx:idx + input_size * hidden_size].reshape(input_size, hidden_size)
    idx += input_size * hidden_size
    b1 = theta[idx:idx + hidden_size]
    idx += hidden_size
    W2 = theta[idx:idx + hidden_size * output_size].reshape(hidden_size, output_size)
    idx += hidden_size * output_size
    b2 = theta[idx:]
    
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_z2 / exp_z2.sum(axis=1, keepdims=True)
    return probs

# Loss function
def compute_loss(theta, X_batch, y_batch):
    probs = forward(X_batch, theta)
    N = X_batch.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), y_batch])
    return np.mean(log_likelihood)

# SPSA Training with the `spsa` library
def train_spsa_with_library(X_train, y_train, num_iterations=num_iterations, batch_size=100):
    print(f"starting the optimization for {num_iterations} steps")
    # Random initialization of parameters
    theta = np.random.randn(theta_size) * 0.01

    loss_history = []

    # Objective function for SPSA (must return scalar loss)
    def objective(theta):
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        loss = compute_loss(theta, X_batch, y_batch)
        loss_history.append(loss)
        return loss

    # # Initialize SPSA optimizer
    # (result, x_history) = spsa.minimize(objective, theta,
    
    #     iterations=num_iterations,
    #     epsilon=1e-8,     # Tolerance for convergence
    # )
    result, x_history = gradient_descent(
        theta, 0.01, 
        lambda x: spsa_gradient(x, objective, 0.01))

    # Run optimization
    #result = optimizer.minimize()
    return result, x_history

# Predict function
def predict(theta, X):
    probs = forward(X, theta)
    return np.argmax(probs, axis=1)

# Evaluate model
def evaluate_spsa(theta, X_test, y_test):
    y_pred = predict(theta, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy (SPSA): {accuracy:.4f}")

# Run training and evaluation
theta, x_history = train_spsa_with_library(X_train, y_train, num_iterations=num_iterations)
evaluate_spsa(theta, X_test, y_test)
compute_loss_and_plot(x_history, X_train, X_test, compute_loss, batch_size=300)