JV
Malladi Tejasvi, CS23M036

The Vision:

    This repo, has my implementation of Neural Nets, with customisable number of layers and layers per neuron and activations(sigmoid, tanh, ReLu). The implementation includes forward pass and backpropogation. The implementation would assume a classification task, with cross entropy loss, this assumption simplifies the computation of the gradients. Yet, Mean Squared error loss would also be implemented for the comparision puropose.

    Also, optimisers including Stochastic Gradient Descent, Momentum Based Gradient Descent, Nesterov accelerated Gradient decent, adam and nadam.

    Experiments would be run on Fashion MNIST dataset, with various hyperparameter combinations with the help of wandb.ai to study and understand the performance.

The Approach:

    Each of the above components would be implemented in an incremental fashion. The python notebook file "JV CS23M036 ASSIGNMENT1.ipynb" would by and large me used to incrementally build the code and unit test it. Once a functionality is built, it would be added to the main code base.

