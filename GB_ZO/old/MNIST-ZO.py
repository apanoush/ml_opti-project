import numpy as np
from keras.datasets import mnist

# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Neural Network Architecture
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

# SPSA training loop
def train_spsa(X_train, y_train, num_iterations=1000, batch_size=64):
    # Hyperparameters for SPSA
    a, A, alpha, gamma, c = 0.16, 100, 0.602, 0.101, 0.26
    theta = np.random.randn(theta_size) * 0.01  # Random initialization
    
    for i in range(num_iterations):
        # Sample mini-batch
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        # Generate perturbation vector
        p = np.random.choice([-1, 1], size=theta.shape)
        delta = c / ((i + 1) ** gamma)
        theta_plus = theta + delta * p
        theta_minus = theta - delta * p
        
        # Compute losses
        loss_plus = compute_loss(theta_plus, X_batch, y_batch)
        loss_minus = compute_loss(theta_minus, X_batch, y_batch)
        
        # Gradient estimate
        gradient_estimate = (loss_plus - loss_minus) / (2 * delta) * p
        
        # Update parameters
        learning_rate = a / (A + i + 1) ** alpha
        theta = theta - learning_rate * gradient_estimate
        
        # Print progress
        if (i + 1) % 100 == 0:
            loss = compute_loss(theta, X_batch, y_batch)
            print(f"Iteration {i+1}, Loss: {loss:.4f}")
    
    return theta

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
theta = train_spsa(X_train, y_train, num_iterations=1000)
evaluate_spsa(theta, X_test, y_test)