- How well do zero-order optimization methods do for ML applications, compared to standard first-order
    - comparison for simple and more complex algorithms (GD and other variants?) for simple or complex problems (different dimensions)
    - analysis of convergence and performance (number of steps)
- Meta-Learning: Can you learn the learning rate? The importance of each datapoint? The direction or
curvature?
1) Effect of gradient clipping on optimization:
   - Does gradient clipping help finding good local minima? 
   - How sensitive are different optimizers to gradient clipping?
   - Is there a relation between the model size and the performance of gradient clipping?
2) Impact of Activation Functions
    - KAN: Kolmogorov-Arnold Networks: "promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”)"
        - https://arxiv.org/abs/2404.19756v1
    - https://arxiv.org/abs/2407.16674v1
    - more interpretable but 10x slower to train
    - implement and analysis of performance on simple tasks vs NLP ? training cost ? accuracy and f1-score? (from other paper KAN are better in "symbol tasks" but worse in more traditional tasks: CV, NLP, ...)

