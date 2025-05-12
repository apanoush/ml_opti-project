- How well do zero-order optimization methods do for ML applications, compared to standard first-order
    - comparison for simple and more complex algorithms (GD and other variants, binary search?) for simple or complex problems (different dimensions: https://openreview.net/forum?id=Skep6TVYDB)
    - analysis of convergence and performance (number of steps)
    - applications in ML and Signal Proc: https://arxiv.org/pdf/2006.06224
    - other related paper: https://openreview.net/forum?id=n1bLgxHW6jW
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

