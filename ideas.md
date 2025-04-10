# anoush

- How does the order in which we see the training data influence the result?
- How well do zero-order optimization methods do for ML applications, compared to standard first-order
methods 
- Meta-Learning: Can you learn the learning rate? The importance of each datapoint? The direction or
curvature?

supplémentaires:
- Fairness-Aware Optimization
    - Modify optimizers to enforce fairness constraints (e.g., demographic parity) during training
    - Can adaptive optimizers (e.g., Adam) be tuned to reduce bias amplification?
    - How do fairness-constrained optimizers trade off accuracy and equity?
- Impact of Activation Functions on Optimization Dynamics
    - Compare optimizer performance (e.g., Adam vs. SGD) across architectures using different activation functions (ReLU, Swish, GeLU)
    - Do smooth activations (e.g., Swish) enable faster convergence for adaptive optimizers?
    - Do certain activations amplify or mitigate issues like gradient saturation?
