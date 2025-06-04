import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Generate synthetic data: y = 2x + 3 + noise
n = 1000
x = np.linspace(0, 1, n)
y_true = 2 * x + 3
y = y_true + np.random.normal(0, 0.5, size=n)

# Loss function (MSE)
def mse_loss(w, b, x, y):
    y_pred = w * x + b
    return np.mean((y_pred - y) ** 2)

# Analytical gradient (Classical GD)
def analytical_gradient(w, b, x, y):
    y_pred = w * x + b
    dw = 2 * np.mean((y_pred - y) * x)
    db = 2 * np.mean(y_pred - y)
    return np.array([dw, db])

# Zero-order gradient (4-point method with orthogonal directions)
def zero_order_gradient(w, b, h, x, y):
    # Generate random direction and its orthogonal
    z1 = np.random.randn(2)
    z1 /= np.linalg.norm(z1)
    z2 = np.array([-z1[1], z1[0]])  # Orthogonal vector
    
    grad_ests = []
    for z in [z1, z2]:
        # Perturbation vector
        delta = h * z
        
        # Points: theta ± delta
        w_plus, b_plus = [w, b] + delta
        w_minus, b_minus = [w, b] - delta
        
        # Evaluate loss at perturbed points
        f_plus = mse_loss(w_plus, b_plus, x, y)
        f_minus = mse_loss(w_minus, b_minus, x, y)
        
        # Gradient estimate for this direction
        grad_dir = z * (f_plus - f_minus) / (2 * h)
        grad_ests.append(grad_dir)
    
    return np.mean(grad_ests, axis=0)

# GD parameters
lr = 0.1          # Learning rate
iterations = 500  # Number of iterations
h_values = [0.01, 0.1, 1.0]  # Perturbation sizes to test
trials = 5        # Number of trials for ZO-GD

# Store results
history = {
    'gd': {'loss': np.zeros(iterations), 'w': [], 'b': []},
    'zo': {h: {'loss': np.zeros((trials, iterations)), 
              'w': np.zeros((trials, iterations)),
              'b': np.zeros((trials, iterations))} for h in h_values}
}

# Classical GD (deterministic, run once)
w_gd, b_gd = 0.0, 0.0
for i in range(iterations):
    loss = mse_loss(w_gd, b_gd, x, y)
    history['gd']['loss'][i] = loss
    grad = analytical_gradient(w_gd, b_gd, x, y)
    w_gd -= lr * grad[0]
    b_gd -= lr * grad[1]
history['gd']['w'].append(w_gd)
history['gd']['b'].append(b_gd)

# ZO-GD (multiple trials for each h)
for h in h_values:
    for trial in range(trials):
        w_zo, b_zo = 0.0, 0.0
        for i in range(iterations):
            loss = mse_loss(w_zo, b_zo, x, y)
            history['zo'][h]['loss'][trial, i] = loss
            grad_est = zero_order_gradient(w_zo, b_zo, h, x, y)
            w_zo -= lr * grad_est[0]
            b_zo -= lr * grad_est[1]
            history['zo'][h]['w'][trial, i] = w_zo
            history['zo'][h]['b'][trial, i] = b_zo

# Analysis: Final parameters and loss
print("Classical GD Final Parameters:")
print(f"  w = {w_gd:.4f}, b = {b_gd:.4f}, Loss = {mse_loss(w_gd, b_gd, x, y):.6f}\n")

for h in h_values:
    avg_w = np.mean([history['zo'][h]['w'][trial, -1] for trial in range(trials)])
    avg_b = np.mean([history['zo'][h]['b'][trial, -1] for trial in range(trials)])
    avg_loss = np.mean([history['zo'][h]['loss'][trial, -1] for trial in range(trials)])
    print(f"ZO-GD (h={h}) Averaged Final Parameters:")
    print(f"  w = {avg_w:.4f}, b = {avg_b:.4f}, Loss = {avg_loss:.6f}")

# Plotting
plt.figure(figsize=(12, 8))

# Loss vs. Iterations
plt.subplot(2, 2, 1)
plt.plot(history['gd']['loss'], label='Classical GD', lw=2)
for h in h_values:
    avg_loss = np.mean(history['zo'][h]['loss'], axis=0)
    plt.plot(avg_loss, '--', label=f'ZO-GD (h={h})')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Convergence')
plt.legend()
plt.grid(True)

# Parameter error (Euclidean distance to true parameters [2, 3])
plt.subplot(2, 2, 2)
true_params = np.array([2, 3])
gd_error = [np.linalg.norm([w_gd - 2, b_gd - 3])] * iterations
plt.plot(gd_error, label='Classical GD', lw=2)
for h in h_values:
    errors = np.zeros(iterations)
    for i in range(iterations):
        trial_errors = []
        for trial in range(trials):
            w = history['zo'][h]['w'][trial, i]
            b = history['zo'][h]['b'][trial, i]
            trial_errors.append(np.linalg.norm([w - 2, b - 3]))
        errors[i] = np.mean(trial_errors)
    plt.plot(errors, '--', label=f'ZO-GD (h={h})')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Parameter Error (L2 Norm)')
plt.title('Parameter Error Convergence')
plt.legend()
plt.grid(True)

# Final loss comparison (boxplot)
plt.subplot(2, 2, 3)
final_losses = [history['gd']['loss'][-1]]  # GD final loss
for h in h_values:
    final_losses.append(history['zo'][h]['loss'][:, -1])  # ZO final losses per trial
plt.boxplot(final_losses, labels=['GD'] + [f'ZO (h={h})' for h in h_values])
plt.ylabel('Final Loss (MSE)')
plt.yscale('log')
plt.title('Final Loss Distribution (5 Trials)')
plt.grid(True)

# Final parameter error comparison (boxplot)
plt.subplot(2, 2, 4)
final_errors = [np.linalg.norm([w_gd - 2, b_gd - 3])]  # GD error
for h in h_values:
    h_errors = []
    for trial in range(trials):
        w = history['zo'][h]['w'][trial, -1]
        b = history['zo'][h]['b'][trial, -1]
        h_errors.append(np.linalg.norm([w - 2, b - 3]))
    final_errors.append(h_errors)
plt.boxplot(final_errors, labels=['GD'] + [f'ZO (h={h})' for h in h_values])
plt.ylabel('Final Parameter Error (L2 Norm)')
plt.yscale('log')
plt.title('Final Parameter Error Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()



def plot_regression_lines(x, y, w_gd, b_gd, w_zo, b_zo, true_w=2, true_b=3):
    """
    Plots the data and regression lines from classical GD and zero-order GD.
    
    Parameters:
    x (np.ndarray): Input features
    y (np.ndarray): Target values
    w_gd (float): Weight from classical GD
    b_gd (float): Bias from classical GD
    w_zo (float): Weight from zero-order GD
    b_zo (float): Bias from zero-order GD
    true_w (float): True weight value
    true_b (float): True bias value
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, alpha=0.6, label='Data points', c='gray')
    
    # Generate predictions for visualization
    x_range = np.linspace(x.min(), x.max(), 100)
    
    # True relationship line
    y_true = true_w * x_range + true_b
    plt.plot(x_range, y_true, 'g-', linewidth=3, label='True: $y=2x+3$')
    
    # Classical GD line
    y_gd = w_gd * x_range + b_gd
    plt.plot(x_range, y_gd, 'b--', linewidth=2.5, 
             label=f'Classical GD: $y={w_gd:.3f}x + {b_gd:.3f}$')
    
    # Zero-order GD line
    y_zo = w_zo * x_range + b_zo
    plt.plot(x_range, y_zo, 'r-.', linewidth=2.5, 
             label=f'ZO-GD: $y={w_zo:.3f}x + {b_zo:.3f}$')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression Line Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Extract parameters for ZO-GD (using best h=0.1 and average of trials)
h_best = 0.1
avg_w_zo = np.mean([history['zo'][h_best]['w'][trial, -1] for trial in range(trials)])
avg_b_zo = np.mean([history['zo'][h_best]['b'][trial, -1] for trial in range(trials)])

# Plot results
plot_regression_lines(
    x=x, 
    y=y,
    w_gd=history['gd']['w'][0],  # From classical GD
    b_gd=history['gd']['b'][0],
    w_zo=avg_w_zo,              # Averaged ZO-GD result
    b_zo=avg_b_zo
)