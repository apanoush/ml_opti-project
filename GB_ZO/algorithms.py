"""script containing all relevant algorithms for linear regression and soft kmeans problems"""

import numpy as np
from tqdm import tqdm

def multipoint_gradient_estimator(x: np.ndarray, function, K: float, random_K_amplificator: bool = False) -> np.ndarray:
    """MPGE using orthogonal perturbations"""

    M = 3 / 2  # amplification range
    random_K_amplificator = np.random.uniform(1 / M, M) if random_K_amplificator else 1

    dimension = x.shape[0]

    # generate first vector
    z = []
    z_first = np.random.randn(dimension)
    z_first /= np.linalg.norm(z_first)
    z.append(z_first)

    # generate remaining orthogonal vectors
    for i in range(1, dimension):
        z_new = np.random.randn(dimension)
        for j in range(i):
            proj = np.dot(z_new, z[j])
            z_new -= proj * z[j]
        norm = np.linalg.norm(z_new)
        if norm > 1e-10:  # avoid division by zero
            z_new /= norm
            z.append(z_new)

    zero_order_estimations = []

    for z_ in z:
        perturbation = z_ * K * random_K_amplificator
        x_plus = x + perturbation
        x_minus = x - perturbation
        f_plus = function(x_plus)
        f_minus = function(x_minus)

        grad_estimate = z_ * (f_plus - f_minus) / (2 * K * random_K_amplificator)
        zero_order_estimations.append(grad_estimate)

    grad = np.mean(zero_order_estimations, axis=0)
    return grad


def spsa_gradient(x: np.ndarray, function, K: float):
    """Simultaneous Perturbation Stochastic Approximation gradient"""

    dimension = x.shape[0]

    # random perturbations vector
    delta = np.random.choice([-1, 1], size=dimension)

    x_plus = x + K * delta
    x_minus = x - K * delta

    f_plus = function(x_plus)
    f_minus = function(x_minus)

    return delta * (f_plus - f_minus) / (2 * K)


def gradient_descent(initial_point, learning_rate, gradient_function, max_iterations=1000, tolerance=1e-6):
    """performs gradient descent optimization"""

    x = initial_point.copy()
    x_history = np.zeros((max_iterations + 1, *x.shape))
    x_history[0] = x

    for i in tqdm(range(1, max_iterations + 1), desc="Performing gradient descent", unit="iteration",
                  total=max_iterations):
        grad = gradient_function(x)

        if tolerance and np.linalg.norm(grad) < tolerance:
            break

        x = x - learning_rate * grad
        x_history[i] = x

    return x, x_history
