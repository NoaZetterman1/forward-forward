from layer import ForwardLayer
from classification_layer import ClassificationLayer
from forward_forward_neutral import ForwardForwardNeutral


import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from datasets import MnistDataset, MnistDataset2, ImdbReviewsDataset

from tqdm import tqdm


from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

import wandb

import warnings


def init_and_train():
    wandb.init()
    batch_size = wandb.config.batch_size
    #epochs = wandb.config.epochs
    forward_forward_lr_layer_0 = 0.001 #wandb.config.lr_ff #0.0006012012836202083 #wandb.config.ff_lr #0.0005
    forward_forward_threshold_layer_0 = 1
    #forward_forward_lr = wandb.config.ff_lr
    #threshold = wandb.config.threshold_multiplier
    optimier_name = "Adam" #wandb.config.optimizer
    weight_decay = 0 #wandb.config.weight_decay_ff #0.002615961100641849 #wandb.config.weight_decay


    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')

    print("Cores in use:", torch.get_num_threads())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using device:", device)
    print()

    transform = Compose(
            [
                ToTensor(),
            ]
        )

    # MNIST
    mnist = torchvision.datasets.MNIST(
                "datasets",
                train=True,
                download=True,
                transform=transform,
            )

    dataset = MnistDataset2(mnist)
    
    
    mnist_test = torchvision.datasets.MNIST(
                "datasets",
                train=False,
                download=True,
                transform=transform,
            )
    test_dataset = MnistDataset2(mnist_test)



    train = dataset

    #train, val = random_split(dataset, [50000, 10000])


    network = ForwardForwardNeutral(device) # FFNeutral slower but better convergence
    l1 = ForwardLayer(input_size=784, output_size=2000, threshold_multiplier=forward_forward_threshold_layer_0, 
                      learning_rate=forward_forward_lr_layer_0, optimizer=optimier_name, weight_decay=weight_decay, device=device)
    l2 = ForwardLayer(input_size=2000, output_size=2000, threshold_multiplier=forward_forward_threshold_layer_0, 
                      learning_rate=forward_forward_lr_layer_0, optimizer=optimier_name, weight_decay=weight_decay, device=device)
    l3 = ForwardLayer(input_size=2000, output_size=2000, threshold_multiplier=forward_forward_threshold_layer_0, 
                      learning_rate=forward_forward_lr_layer_0, optimizer=optimier_name, weight_decay=weight_decay, device=device)
    l4 = ForwardLayer(input_size=2000, output_size=2000, threshold_multiplier=forward_forward_threshold_layer_0, 
                      learning_rate=forward_forward_lr_layer_0, optimizer=optimier_name, weight_decay=weight_decay, device=device)

    network.add_layer(l1)
    network.add_layer(l2)
    network.add_layer(l3)
    network.add_layer(l4)

    #network.add_classification_layer(ClassificationLayer(input_size=2000, output_size=10, optimizer="Adam", learning_rate=wandb.config.lr_ff_classification, weight_decay=wandb.config.weight_decay_ff_classification, device=device))
    network.add_classification_layer(ClassificationLayer(input_size=2000, output_size=10, optimizer="Adam", learning_rate=0.01, weight_decay=0, device=device))

    #network.train_ff(train, test_dataset, batch_size, epochs=wandb.config.epochs_ff)
    network.train_backprop(train, test_dataset, batch_size, epochs=wandb.config.epochs_backprop, learning_rate=wandb.config.lr_backprop, weight_decay=wandb.config.weight_decay_backprop)

    

#TODO: Use multiclassification

# https://wandb.ai/forward-forward-project/uncategorized/runs/0wumqcj6/overview?workspace=user-n-zetterman
# Optimal layer 0-4 params

#Maybe optimal class layer params with the above previous layer params
#https://wandb.ai/forward-forward-project/uncategorized/runs/6lyll2ab/overview?workspace=user-n-zetterman

wandb.login(key="535f4a320d7e236eafb539ae54a1adff047d97fc")

# Run this over each layer in F-F And optimize greedily (otherwise too much time )

""" BEST SGD PARAMS
'parameters_l1': 
{   
    'batch_size': {'values': [32, 64, 128, 256]},
    'epochs': {'values': [30]},
    'ff_lr': {'values': [0.0005]},
    'threshold_multiplier': {'values':[1]},
    #'optimizer': {'values': [optim.SGD, optim.Adam]},
    'weight_decay': {'values':[3e-4]}
    },

"""

#ADAM > SGD (more stable, about same convergence)
# ADAM LR: 0.0005
#SGD LR: 0.0005
# batch size of 32 or 64 best. (but smaller takes longer)
#overall optimal LR seem to be 0.0005

"""
sweep_configuration = {
     
    'batch_size': 128,
    'epochs_ff': 30,
    'epochs_backprop': 15,
    'lr_backprop': 0.000045,
    'lr_ff': 0.0005,
    'lr_ff_classification': 0.035,
    'weight_decay_ff': 0.000017,
    'weight_decay_ff_classification': 0.083,
    'weight_decay_backprop': 0.00001
     
}"""

#FF Sweeo configuration
"""
sweep_configuration = {
    'method': 'bayes',
    'name': "ff_sweep",
    'metric': {'name': 'combined_score', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [128]},
        'epochs_ff': {'values': [60]},
        'lr_ff': {'distribution': 'log_uniform_values', 'max':0.1, 'min': 0.000001},
        'lr_ff_classification': {'distribution': 'log_uniform_values', 'max':0.1, 'min': 0.000001},
        'weight_decay_ff': {'distribution': 'log_uniform_values', 'max': 0.1, 'min': 0.000001},
        'weight_decay_ff_classification': {'distribution': 'log_uniform_values', 'max': 0.1, 'min': 0.000001}
    }
}

sweep_configuration = {
    'method': 'bayes',
    'name': "ff_sweep",
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [128]},
        'epochs_backprop': {'values': [15, 20, 25]},
        'lr_backprop': {'distribution': 'log_uniform_values', 'max':0.1, 'min': 0.000001},
        'weight_decay_backprop': {'distribution': 'log_uniform_values', 'max': 0.1, 'min': 0.000001},
    }
}"""



#sweep_id = wandb.sweep(sweep_configuration)

#wandb.agent(sweep_id, function=init_and_train)
#init_and_train()
# Store stuff in wandb
run_configuration = {
     
    'batch_size': 128,
    'epochs_backprop': 15,
    'lr_backprop': 0.00026957312465981313,
    'weight_decay_backprop': 0.00000277043594564226
}

for i in range(100):
    run = wandb.init(
    #mode="disabled",
    project="ForwardForward",
    #entity  = "automellon",
    name = "BP15", # Wandb creates random run names if you skip this field
    reinit = True, # Allows reinitalizing runs when you re-run this cell
    # run_id = # Insert specific run id here if you want to resume a previous run
    # resume = "must" # You need this to resume previous runs, but comment out reinit = True when using this
    config = run_configuration
    )


    init_and_train()





