JV
Malladi Tejasvi, CS23M036

> The train.py file has the modules for Activation Functions, Initalizers, Neural Network, Optimisers and other supporting methods/modules.

> train.py could be used to import these modules to use elsewhere, like WandBSweep.py does.

> It could also be invoked using the following commandline args:

           Name	                Default Value	Description
           
    -wp,    --wandb_project    	None	        Project name used to track experiments in Weights & Biases dashboard
    -we,    --wandb_entity	    None	        Wandb Entity used to track experiments in the Weights & Biases dashboard.
    -d,     --dataset	        fashion_mnist	Choices: ["mnist", "fashion_mnist"]
    -e,     --epochs	        25	            Number of epochs to train neural network.
    -b,     --batch_size	    32	            Batch size used to train neural network.
    -l,     --loss	            cross_entropy	choices: ["mean_squared_error", "cross_entropy"]
    -o,     --optimizer	        nadam	        choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    -lr,    --learning_rate	    1e-4            Learning rate used to optimize model parameters
    -m,     --momentum	        0.5	            Momentum used by momentum and nag optimizers.
    -beta,  --beta	            0.5	            Beta used by rmsprop optimizer
    -beta1, --beta1	            0.9             Beta1 used by adam and nadam optimizers.
    -beta2, --beta2	            0.999	        Beta2 used by adam and nadam optimizers.
    -eps,   --epsilon	        1e-8	        Epsilon used by optimizers.
    -w_d,   --weight_decay	    .0	            Weight decay used by optimizers.
    -w_i,   --weight_init	    xavier	        choices: ["random", "Xavier"]
    -nhl,   --num_layers	    4	            Number of hidden layers used in feedforward neural network.
    -sz,    --hidden_size	    64	            Number of hidden neurons in a feedforward layer.
    -a,     --activation	    tanh	        choices: ["identity", "sigmoid", "tanh", "ReLU"]
    >


> Please note that a wandb project name has to be passed to enable logging in wandb.
> The code train.py when invoked from the commandline/terminal takes the appropriate dataset, randomly splits 10% of train data into validation data, creates a neural net, trains it
> and finally computes the test accuracy and generates the confusion matrix plots.

> The python notebook JV CS23M036 ASSIGNMENT1.ipynb, has the code and outputs, including the visualization of the fashion MNIST dataset, training of the model after hyperparameter tuning, test accuracy computation and plotting of confusion matrix. This notebook has been the place where most of the code was developed, this choice allowed rigirous resting of various parts of the codes to verify the correctness.

> V HYPERPARAMETER TUNING WANDB.ipynb again is the notebook which lead to WandBSweep.py
> To perform hyperparameter tuning, WandBSweep.py has to be edited as follows:
    1. Modify the sweep configuration as required.
    2. Add the API key for wandb login
    3. Add the wand project name as required
> good to go!!


The Vision:
    This repo, has my implementation of Neural Nets, with customisable number of layers and layers per neuron and activations(sigmoid, tanh, ReLu). The implementation includes forward pass and backpropogation. The implementation would assume a classification task, with cross entropy loss, this assumption simplifies the computation of the gradients. Yet, Mean Squared error loss would also be implemented for the comparision puropose.

    Also, optimisers including Stochastic Gradient Descent, Momentum Based Gradient Descent, Nesterov accelerated Gradient decent, adam and nadam.

    Experiments would be run on Fashion MNIST dataset, with various hyperparameter combinations with the help of wandb.ai to study and understand the performance.

The Approach:

    Each of the above components would be implemented in an incremental fashion. The python notebook file "JV CS23M036 ASSIGNMENT1.ipynb" would by and large me used to incrementally build the code and unit test it. Once a functionality is built, it would be added to the main code base.

