#JV

from train import Initializer,Activation,Activation_Gradient,NeuralNetwork,Optimiser
from keras.datasets import fashion_mnist,mnist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

#"""WandB API Key Masked"""
wandb.login(key="")

# Creating a wandb sweep config

optim = Optimiser

sweep_config = {
    'method': 'grid',
    'name' : 'sweep cross entropy and MSE',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'number_of_hidden_layers': {
            'values': [4,5]
        },    
         'hidden_size':{
            'values':[64]
        },
        'activation': {
            'values': ['tanh']
        },
        
        'initialization': {
            'values': ["xavier"]

        },
        'optimiser': {
            'values': ["nadam"]
        },
        
        'epochs': {
            'values': [25]
        },

        'batch_sizes': {
            'values': [32,64]
        },
        
        'lr': {
            'values': [1e-4]
        },
        'weight_decay': {
            'values': [0.5]
        },
        'loss': {
            'values': ['cross_entropy','mean_squared_error']
        },


    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='JV_CS23M036_TEJASVI_DL_ASSIGNMENT1')


## Data Split and Normalisation

## train-test data is got from the mnist fashion import data call
## now this data is being flattened, i.e each image into a 1d array
## then flattened train data is split into 80% train and 20% validation data.

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


seed = 76 #setting this as seed wherever randomness comes

x_train_flattened = x_train.flatten().reshape(x_train.shape[0],-1)/255
x_test_flattened = x_test.flatten().reshape(x_test.shape[0],-1)/255


#np.random.seed(seed)
random_num_generator = np.random.RandomState(seed)


validation_indices = random_num_generator.choice(x_train_flattened.shape[0],int(0.1*x_train_flattened.shape[0]),replace=False)
train_indices = np.array(list(set(np.arange(x_train_flattened.shape[0])).difference(set(validation_indices))))

x_train_data = x_train_flattened[train_indices]
y_train_data = y_train[train_indices]

x_validation_data = x_train_flattened[validation_indices]
y_validation_data = y_train[validation_indices]

optimiser_params_dict = {"momentum":0.9,"beta":0.5,"beta1":0.9,"beta2":0.999,"epsilon":1e-8}

def create_nnet_and_train(config):
    ##create a neural network.
    
    
    nn = NeuralNetwork(seed=seed)
    
    
    number_of_hidden_layers = config['number_of_hidden_layers']
    neurons_per_hidden_layer=[config['hidden_size']]*number_of_hidden_layers ## Assuming all layers have same number of neurons
    
    initialization = config['initialization']
    
    
    activation = config['activation']
    
    
    ## Create NNet with the current architecture config
    nn.createNNet(number_of_hidden_layers=number_of_hidden_layers,neurons_per_hidden_layer=neurons_per_hidden_layer,initialization = initialization,activation=activation)
    
    optim = Optimiser()
    
    
    loss_type=config['loss']
    optimiser_algo = config['optimiser']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_sizes']
    l2_param = config['weight_decay']
    
    ##train NNet.
    train_data = [x_train_data,y_train_data]

    val_data = [x_validation_data,y_validation_data]

    print_val_accuracy = True

    log_wandb_data = True


    optim.train(nn,train_data,val_data,optimiser_algo,optimiser_params_dict,lr,epochs,batch_size,l2_param,print_val_accuracy,loss_type,log_wandb_data)

def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''


    with wandb.init() as run:

        run_name="-hl_"+str(wandb.config.number_of_hidden_layers)+"-hs_"+str(wandb.config.hidden_size)+"-init_"+wandb.config.initialization+"-ac_"+wandb.config.activation

        run_name = run_name+"-optim_"+str(wandb.config.optimiser)+"-lr_"+str(wandb.config.lr) +"-epochs_"+str(wandb.config.epochs)+"-bs_"+str(wandb.config.batch_sizes)+"-reg_"+str(wandb.config.weight_decay)

        wandb.run.name=run_name

        create_nnet_and_train(wandb.config)
        

wandb.agent(sweep_id, function=main,count=100) # calls main function for count number of times.
wandb.finish()