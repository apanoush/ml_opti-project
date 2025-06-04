import numpy as np

def multipoint_gradient_estimator(x: np.ndarray, function, K: int, random_K_amplificator: bool=False) -> np.ndarray:
    """differentiation of the F function, with respect to x"""
    M = 3/2#5
    random_K_amplificator = np.random.uniform(1/M, M) if random_K_amplificator else 1

    dimension = x.shape[0]

    # Generate orthogonal vectors for each dimension
    z = []

    # First vector: random unit vector
    z_first = np.random.randn(dimension)
    z_first /= np.linalg.norm(z_first)
    z.append(z_first)

    # Generate remaining orthogonal vectors using Gram-Schmidt process
    for i in range(1, dimension):
        # Start with a random vector
        z_new = np.random.randn(dimension)
        
        # Orthogonalize against all previous vectors
        for j in range(i):
            proj = np.dot(z_new, z[j])
            z_new = z_new - proj * z[j]
        
        # Normalize to unit length
        norm = np.linalg.norm(z_new)
        if norm > 1e-10:  # Avoid division by zero
            z_new /= norm
            z.append(z_new)

        zero_order_estimations = []

        for z_ in z:

            perturbation = z_ * K * random_K_amplificator
            # we recompute GD for new x+perturbation
            x_plus = x + perturbation
            x_minus = x - perturbation
            zero_order_estimation = z_ * (function(x_plus) - function(x_minus)) / (2 * K + 1e-5)
            zero_order_estimations.append(zero_order_estimation)

    return np.array(zero_order_estimations).mean(axis=0)


# SPSA gradient estimation
def spsa_gradient(x: np.ndarray, function, K: int):
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