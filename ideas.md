- How well do zero-order optimization methods do for ML applications, compared to standard first-order
    - comparison for simple and more complex algorithms (GD and other variants, proximal GD) for simple or complex problems (different dimensions: https://openreview.net/forum?id=Skep6TVYDB)
    - analysis of convergence and performance (number of steps)
    - applications in ML and Signal Proc: https://arxiv.org/pdf/2006.06224
    - other related paper: https://openreview.net/forum?id=n1bLgxHW6jW
    - multiple ZOO methods (n-point optimization, with randomness, ...)
    - examples: Chance Constrained Optimization, Integer Linear Programming, hyperparemeter optimizaition 
        - linear regression with 2 parameters: fit $y=w\cdot x + b$ to synthetic data
            - GB: MSE, ZOO: coordinate descent
        - kmeans as an optimization problem:
            - GB: probabilistic clustering, soft assignment formula (fuzzy clustering: derivable) ([source](https://en.wikipedia.org/wiki/Fuzzy_clustering)): $\sum_{i=1}^{n} \sum_{j=1}^{c} w_{ij}^m \left\|\mathbf{x}_i - \mathbf{c}_j \right\|^2$ 
            - ZO: coordinate descent (bad?) or random search (pertubrs on centroid randomly and accepts if loss descreases)
        - Neural Network for MNIST Digit Classification
            - GB: classical backpropagation
            - ZO: Perturb each parameter slightly and approximate gradients
                - SPSA: perturb all parameters at once with a random vector to estimate gradients efficiently 
                    - https://www.jhuapl.edu/SPSA/Pages/
                    - https://github.com/anomic1911/SPSA-Net
- Meta-Learning: Can you learn the learning rate? The importance of each datapoint? The direction or
curvature?

Learned optimizers: 
Good summary: "Classic optimization methods are built upon components that are basic methods — such as gradient descent, conjugate gradient, Newton steps, Simplex basis update, and stochastic sampling — in a theoretically justified manner. Most conventional optimization methods can be written in a few lines, and their performance is guaranteed by their theories.""L2O is an alternative paradigm that develops an optimization method by training, i.e., learning from its performance on sample problems. The method may lack a solid theoretical basis but improves its performance during the training process."
Key idea : instead of writing the optimizers as a classic function replace it (or part of of it) by a neural network that can be learned on sample tasks.
Compared its performance on unseen task compared to classical methods such as SGD or ADAM.
Difficulty of generalization on different tasks, thus better to choose a specific type of task.
papers experimenting : https://arxiv.org/pdf/2312.07174, https://arxiv.org/pdf/2103.12828, https://arxiv.org/pdf/2405.18222
Need to find : case study 

- Effect of gradient clipping on optimization:
    Theory behind it: https://arxiv.org/pdf/1905.11881, https://arxiv.org/pdf/2305.01588 (EPFL, more complicated)
   - Main idea: can we improve the convergence behaviour of some algorithms that require gradient clipping (like SGD for RNN) by predicting the future gradients. For example gradients could be increasing while still not exploding, so we do not want to clip them in this case.
     Potential paper: https://arxiv.org/pdf/2504.02507, but does not seem to really predict the future gradients.
   - Other ideas, more general:
     - Does gradient clipping help finding good local minima?
     - How sensitive are different optimizers to gradient clipping?
     - Is there a relation between the model size and the performance of gradient clipping?
- Impact of Activation Functions
    - KAN: Kolmogorov-Arnold Networks: "promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”)"
        - https://arxiv.org/abs/2404.19756v1
    - https://arxiv.org/abs/2407.16674v1
    - more interpretable but 10x slower to train
    - implement and analysis of performance on simple tasks vs NLP ? training cost ? accuracy and f1-score? (from other paper KAN are better in "symbol tasks" but worse in more traditional tasks: CV, NLP, ...)

