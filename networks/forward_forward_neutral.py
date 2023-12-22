import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import wandb

class ForwardForwardNeutral():
    """
    ForwardForwardNeutral is a class for training and evaluating a neural network with multiple forward layers
    followed by a classification layer connected to the last classification layer. The classification layer is trained by 
    sending an image with a neutral label through the network, as opposed to the forward layers that are trained with positive and negative labels
    """

    def __init__(self, device):

        self.forward_layers = []
        self.classification_layer = None
        self.device = device
    
    def forward(self, sample, freeze: []):
        input = torch.cat([sample['pos']['image'], sample['neg']['image']]).to(self.device)
        binary_labels = torch.cat([sample['pos']['binary_label'], sample['neg']['binary_label']]).to(self.device)
        labels = torch.cat([sample['pos']['label'], sample['neg']['label']]).to(self.device)
        
        # Train F-F Layers
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward(input, binary_labels, freeze[i])

        # Train classification layers
        input = sample['neutral']['image'].to(self.device)
        binary_labels = sample['neutral']['binary_label'].to(self.device)
        labels = sample['neutral']['label'].to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward(input, binary_labels, freeze=True)
        
        output = self.classification_layer.forward(input, labels)

        return output
    
    def add_layer(self, layer):
        self.forward_layers.append(layer)
    
    def add_classification_layer(self, layer):
        self.classification_layer = layer

    def train(self, train, validation, batch_size=128, epochs=1):
        """
        Train the neural network.

        Parameters
        ----------
        train: Dataset
            The training dataset.
        validation: Dataset
            The validation dataset.
        batch_size: int
            The training batch size to use (default is 128)
        epochs: int
            Number of training epochs (default is 1).

        Returns
        -------
            None
        """

        freeze = [False] * (len(self.forward_layers)+1) # May want to freeze early layers after some training

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
        validation2_dataloader = DataLoader(train, batch_size=len(train), num_workers=16, persistent_workers=True)
        validation_dataloader = DataLoader(validation, batch_size=len(validation), num_workers=16, persistent_workers=True)
        
        for epoch in tqdm(range(epochs)):

            for sample in train_dataloader:

                self.forward(sample, freeze)

            self.evaluate(validation_dataloader, epoch, "val")
            self.evaluate(validation2_dataloader, epoch, "train")

            #if epoch == 50:
            #    freeze = [False] * (len(self.forward_layers)+1)

            #if epoch==10:
            #    freeze[0] = True

    def evaluate(self, validation_dataloader, epoch, partition):
        data = next(iter(validation_dataloader))
        
        input = torch.cat([data['pos']['image'], data['neg']['image']]).to(self.device)
        binary_labels = torch.cat([data['pos']['binary_label'], data['neg']['binary_label']]).to(self.device)
        labels = torch.cat([data['pos']['label'], data['neg']['label']]).to(self.device)

        wandb_logging = {}
        input1=None
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
            input1 = input

            sum_squares = torch.sum(input1**2, dim=1)
            
                
            correct = torch.where(binary_labels==0, sum_squares<=2000, sum_squares>2000)
            wandb_logging[f"{partition}_layer_{i}_acc"] = sum(correct==True)/len(labels)
            
            layer.step_update(epoch)
            
        input = data['neutral']['image'].to(self.device)
        labels = data['neutral']['label'].to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
        
        output = self.classification_layer.eval(input)
        self.classification_layer.step_update(epoch)
        
        predictions = torch.argmax(output, axis=1)
        wandb_logging[f"{partition}_prediction_acc"] = torch.sum((predictions==labels))/len(labels)

        wandb.log(wandb_logging, step=epoch)


    def test(self):
        pass