from layers.classification_layer import ClassificationLayer
from layers.recurrent_forward_layer import RecurrentForwardLayer
from networks.forward_forward_recurrent import ForwardForwardRecurrent
import os

import numpy as np
import pandas as pd

import itertools
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from datasets import MnistDataset, MnistDatasetTriplet, ImdbReviewsDataset, ToyReviewDataset, ImdbReviewDatasetOneHot

from tqdm import tqdm


from torchvision.transforms import Compose, ToTensor

import wandb

# Store runs in wandb
run = wandb.init(
mode="disabled",
project="ForwardForward",
entity  = "forward-forward-project",
#name = "FF-recurrent-imdbtest-adam", # Skip this to generate a random run name
reinit = False,
)


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

#dataset = MnistDataset2(mnist)

#train_df = pd.read_csv("datasets/IMDB/reviews_train_text.csv", names=["review", "sentiment"])#[0:6]
train_df = pd.read_csv("datasets/toy_dataset/sentiments.csv", names=["sentence", "sentiment"])
#dataset = ImdbReviewDatasetOneHot(train_df)
#ataset = ToyReviewDataset(train_df)
#Try to do something with the word dataset
# Split into train/test/val
#print("Dataset created")
#train, val = random_split(dataset, [4, 2], generator=torch.Generator().manual_seed(2))
#train = dataset #, val, _ = random_split(dataset, [20, 1], generator=torch.Generator().manual_seed(2))
# Train:

optimizers = ["Adam"]
normalize_recurrents = [False]
recurrent_sizes = [100]
pre_trained_embeddings = [True]
thresholdss = [1]
activation_fns = [nn.ReLU]
train_df = pd.read_csv("datasets/IMDB/reviews_train_text.csv", names=["review", "sentiment"])
print(type(train_df))
dataset = ImdbReviewsDataset(train_df)
train, val = random_split(dataset, [40000, 10000], generator=torch.Generator().manual_seed(2))
#dataset = ToyReviewDataset(train_df, True)

for optimizer, normalize_recurrent, recurrent_size, pre_trained_embedding, threshold, activation_fn in list(itertools.product(optimizers, normalize_recurrents, recurrent_sizes, pre_trained_embeddings, thresholdss, activation_fns)):
    # Also append label vs not
    #print(f"SGD & {normalize_recurrent} & {recurrent_size} & {pre_trained_embedding} & {threshold} & {activation_fn} & ")
    #print("optimizer=", optimizer, " norm_recurrent=", normalize_recurrent, " recurrent_size=", recurrent_size, " pre_train_emb=", pre_trained_embedding, " threshold=", threshold, "activation_fn=", activation_fn)

    
    #dataset = ImdbReviewDatasetOneHot(train_df)
    

    
    network = ForwardForwardRecurrent(device)
    # First layer learns representations
    #l1 = RecurrentForwardLayer(input_size=110, output_size=30, threshold_multiplier=1, optimizer="SGD", normalize_recurrent=False, activation_fn=nn.ReLU, device=device)
    #l1.set_layernr(0)
    #l1.set_recurrent_connection_from(l1)
    #l1 = ForwardLayer(784, 2000, 1, device=device)
    l2 = RecurrentForwardLayer(300+recurrent_size, recurrent_size, threshold_multiplier=threshold, optimizer=optimizer, normalize_recurrent=normalize_recurrent, activation_fn=activation_fn, device=device)
    l2.set_layernr(0)
    l2.set_recurrent_connection_from(l2)
    #l1.set_recurrent_connection_from(l1)
    #l3 = RecurrentForwardLayer(1000, 500, 1, device=device)
    #l1.set_recurrent_connection(l1)
    #l4 = RecurrentForwardLayer(2000, 2000, 1, device=device)
    #network.add_layer(l1)
    network.add_layer(l2)
    #network.add_layer(l3)
    #network.add_layer(l4)
    network.add_classification_layer(ClassificationLayer(input_size=recurrent_size, output_size=2, device=device))

    network.train(train, val, epochs=10)


