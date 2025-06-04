import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_loss_history(loss_history):
    """Plots the training loss over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration (or Batch)')
    plt.ylabel('Loss')
    plt.title('SPSA Training Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_loss_and_plot(x_history, loss_function, X_train=None, y_train=None, batch_size = None):
    # Plot the loss history
    num_iterations = x_history.shape[0]
    loss_history = np.zeros(num_iterations)
    for i, xh in tqdm(enumerate(x_history), desc="Computing the loss", unit="x", total=num_iterations):
        if batch_size and X_train and y_train:
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_train= X_train[indices]
            y_train = y_train[indices]
            loss_history[i] = loss_function(xh, X_train, y_train)
        else:
            loss_history[i] = loss_function(xh)
    print(f"Final loss: {loss_history[-1]}")
    plot_loss_history(loss_history)