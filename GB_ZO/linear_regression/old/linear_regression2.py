import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
sys.path.insert(0, ".")
import GB_ZO.spsa as spsa

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data: y = 2x + 3 + noise
n = 1000
x = np.linspace(0, 1, n)
y_true = 2 * x + 3
y = y_true + np.random.normal(0, 0.4, size=n)

# Loss function (MSE)
def mse_loss(theta, x, y):
    """Mean Squared Error loss function"""
    w, b = theta
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
        f_plus = mse_loss([w_plus, b_plus], x, y)
        f_minus = mse_loss([w_minus, b_minus], x, y)
        
        # Gradient estimate for this direction
        grad_dir = z * (f_plus - f_minus) / (2 * h)
        grad_ests.append(grad_dir)
    
    return np.mean(grad_ests, axis=0)

# SPSA gradient estimation
def spsa_gradient(theta, c, x, y):
    """Simultaneous Perturbation Stochastic Approximation gradient"""
    # Generate Rademacher-distributed perturbation vector (±1)
    delta = np.random.choice([-1, 1], size=2)
    
    # Perturbed parameters
    theta_plus = theta + c * delta
    theta_minus = theta - c * delta
    
    # Evaluate loss at perturbed points
    f_plus = mse_loss(theta_plus, x, y)
    f_minus = mse_loss(theta_minus, x, y)
    
    # Gradient estimate
    return delta * (f_plus - f_minus) / (2 * c)

# def spsa_gradient(theta, c, x, y):
#     (result, x_history) = spsa.minimize(lambda u: mse_loss(u, x,y), theta,
    
#         iterations=1,
#         epsilon=1e-8,     # Tolerance for convergence
#         silent=True
#     )
#     return result

# GD parameters
lr = 0.1          # Learning rate
iterations = 500  # Number of iterations
h_values = [0.01, 0.1, 1.0]  # Perturbation sizes for ZO-GD
c_values = [0.01, 0.1, 1.0]  # Perturbation sizes for SPSA
trials = 5        # Number of trials for stochastic methods

# Store results
history = {
    'gd': {'loss': np.zeros(iterations), 'params': np.zeros((iterations, 2))},
    'zo': {h: {'loss': np.zeros((trials, iterations)), 
              'params': np.zeros((trials, iterations, 2))} for h in h_values},
    'spsa': {c: {'loss': np.zeros((trials, iterations)), 
                 'params': np.zeros((trials, iterations, 2))} for c in c_values}
}

# Classical GD (deterministic, run once)
theta_gd = np.array([0.0, 0.0])  # [w, b]
for i in range(iterations):
    loss = mse_loss(theta_gd, x, y)
    history['gd']['loss'][i] = loss
    history['gd']['params'][i] = theta_gd
    grad = analytical_gradient(theta_gd[0], theta_gd[1], x, y)
    theta_gd -= lr * grad

# ZO-GD (multiple trials for each h)
for h in h_values:
    for trial in range(trials):
        theta_zo = np.array([0.0, 0.0])  # [w, b]
        for i in range(iterations):
            loss = mse_loss(theta_zo, x, y)
            history['zo'][h]['loss'][trial, i] = loss
            history['zo'][h]['params'][trial, i] = theta_zo
            grad_est = zero_order_gradient(theta_zo[0], theta_zo[1], h, x, y)
            theta_zo -= lr * grad_est

# SPSA (multiple trials for each c)
for c in c_values:
    for trial in range(trials):
        theta_spsa = np.array([0.0, 0.0])  # [w, b]
        for i in range(iterations):
            loss = mse_loss(theta_spsa, x, y)
            history['spsa'][c]['loss'][trial, i] = loss
            history['spsa'][c]['params'][trial, i] = theta_spsa
            grad_est = spsa_gradient(theta_spsa, c, x, y)
            theta_spsa -= lr * grad_est

# Analysis: Final parameters and loss
print("Classical GD Final Parameters:")
print(f"  w = {theta_gd[0]:.4f}, b = {theta_gd[1]:.4f}, Loss = {mse_loss(theta_gd, x, y):.6f}\n")

for h in h_values:
    final_params = np.mean(history['zo'][h]['params'][:, -1, :], axis=0)
    final_loss = np.mean([mse_loss(p, x, y) for p in history['zo'][h]['params'][:, -1, :]])
    print(f"ZO-GD (h={h}) Averaged Final Parameters:")
    print(f"  w = {final_params[0]:.4f}, b = {final_params[1]:.4f}, Loss = {final_loss:.6f}")

for c in c_values:
    final_params = np.mean(history['spsa'][c]['params'][:, -1, :], axis=0)
    final_loss = np.mean([mse_loss(p, x, y) for p in history['spsa'][c]['params'][:, -1, :]])
    print(f"SPSA (c={c}) Averaged Final Parameters:")
    print(f"  w = {final_params[0]:.4f}, b = {final_params[1]:.4f}, Loss = {final_loss:.6f}")

# Plotting regression lines
def plot_regression_lines(x, y, methods, labels, true_w=2, true_b=3):
    """Plots data and regression lines from multiple methods"""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, alpha=0.6, label='Data points', c='gray')
    
    # Generate predictions for visualization
    x_range = np.linspace(x.min(), x.max(), 100)
    
    # True relationship line
    y_true = true_w * x_range + true_b
    plt.plot(x_range, y_true, 'g-', linewidth=3, label='True: $y=2x+3$')
    
    # Colors for methods
    colors = ['b', 'r', 'm', 'c']
    
    # Plot each method's regression line
    for i, (w, b) in enumerate(methods):
        y_pred = w * x_range + b
        plt.plot(x_range, y_pred, linestyle='--', linewidth=2.5, color=colors[i],
                 label=f'{labels[i]}: $y={w:.3f}x + {b:.3f}$')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression Line Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Select best parameters for each method
best_zo = np.mean(history['zo'][0.1]['params'][:, -1, :], axis=0)  # Using h=0.1
best_spsa = np.mean(history['spsa'][0.1]['params'][:, -1, :], axis=0)  # Using c=0.1

# Create plot
plot_regression_lines(
    x=x, 
    y=y,
    methods=[
        (theta_gd[0], theta_gd[1]),         # Classical GD
        (best_zo[0], best_zo[1]),           # ZO-GD
        (best_spsa[0], best_spsa[1])        # SPSA
    ],
    labels=['Classical GD', 'ZO-GD (h=0.1)', 'SPSA (c=0.1)']
)

# Convergence plot comparison
plt.figure(figsize=(12, 6))

# Loss vs. Iterations
plt.subplot(1, 2, 1)
plt.plot(history['gd']['loss'], label='Classical GD', lw=2)

# ZO-GD (h=0.1)
avg_zo_loss = np.mean(history['zo'][0.1]['loss'], axis=0)
plt.plot(avg_zo_loss, '--', label='ZO-GD (h=0.1)')

# SPSA (c=0.1)
avg_spsa_loss = np.mean(history['spsa'][0.1]['loss'], axis=0)
plt.plot(avg_spsa_loss, '-.', label='SPSA (c=0.1)')

plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Convergence')
plt.legend()
plt.grid(True)

# Parameter error (Euclidean distance to true parameters [2, 3])
plt.subplot(1, 2, 2)
true_params = np.array([2, 3])

# Classical GD error
gd_error = [np.linalg.norm(p - true_params) for p in history['gd']['params']]
plt.plot(gd_error, label='Classical GD', lw=2)

# ZO-GD error (h=0.1)
zo_errors = np.zeros(iterations)
for i in range(iterations):
    trial_errors = [np.linalg.norm(p[i] - true_params) 
                    for p in history['zo'][0.1]['params']]
    zo_errors[i] = np.mean(trial_errors)
plt.plot(zo_errors, '--', label='ZO-GD (h=0.1)')

# SPSA error (c=0.1)
spsa_errors = np.zeros(iterations)
for i in range(iterations):
    trial_errors = [np.linalg.norm(p[i] - true_params) 
                    for p in history['spsa'][0.1]['params']]
    spsa_errors[i] = np.mean(trial_errors)
plt.plot(spsa_errors, '-.', label='SPSA (c=0.1)')

plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Parameter Error (L2 Norm)')
plt.title('Parameter Error Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()