import numpy as np
from tqdm import tqdm

def multipoint_gradient_estimator(x: np.ndarray, function, K: float, random_K_amplificator: bool=False) -> np.ndarray:
    """Multipoint gradient estimator using orthogonal perturbations."""

    M = 3/2  # Amplification range
    random_K_amplificator = np.random.uniform(1/M, M) if random_K_amplificator else 1

    dimension = x.shape[0]

    # Generate orthogonal vectors for each dimension

    # First vector: random unit vector
    z = []
    z_first = np.random.randn(dimension)
    z_first /= np.linalg.norm(z_first)
    z.append(z_first)

    # Generate remaining orthogonal vectors using Gram-Schmidt
    for i in range(1, dimension):
        z_new = np.random.randn(dimension)
        for j in range(i):
            proj = np.dot(z_new, z[j])
            z_new -= proj * z[j]
        norm = np.linalg.norm(z_new)
        if norm > 1e-10:  # Avoid division by zero
            z_new /= norm
            z.append(z_new)

    zero_order_estimations = []

    for z_ in z:
        perturbation = z_ * K * random_K_amplificator
        x_plus = x + perturbation
        x_minus = x - perturbation
        f_plus = function(x_plus)
        f_minus = function(x_minus)

        # Gradient estimate along this direction
        grad_estimate = z_ * (f_plus - f_minus) / (2 * K * random_K_amplificator)
        zero_order_estimations.append(grad_estimate)

    # Average over all directions
    grad = np.mean(zero_order_estimations, axis=0)
    return grad

def spsa_gradient(x: np.ndarray, function, K: float):
    """Simultaneous Perturbation Stochastic Approximation gradient"""

    dimension = x.shape[0]

    # Generate Rademacher-distributed perturbation vector (±1)
    delta = np.random.choice([-1, 1], size=dimension)

    # Perturbed parameters
    x_plus = x + K * delta
    x_minus = x - K * delta

    # Evaluate loss at perturbed points
    f_plus = function(x_plus)
    f_minus = function(x_minus)

    # Gradient estimate
    return delta * (f_plus - f_minus) / (2 * K)


def gradient_descent(initial_point, learning_rate, gradient_function, max_iterations=1000, tolerance=1e-6):
    """Performs gradient descent optimization"""
    
    x = initial_point.copy()
    x_history = [x.copy()]
    
    for i in tqdm(range(max_iterations), desc="Performing gradient descent", unit="iteration", total=max_iterations):
        grad = gradient_function(x)
        
        # Check for convergence
        if np.linalg.norm(grad) < tolerance:
            break
            
        # Update parameters
        x = x - learning_rate * grad
        x_history.append(x.copy())
    
    return x, x_history

