import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import wandb

class ForwardForward():
    """
    Simples Forward Forward network with no modifications. Does forward passes with positive and negative samples and updates
    forward layers as well as the classification layer in the same iteration. Does not work well
    """

    def __init__(self, device):

        self.forward_layers = []
        self.classification_layer = None
        self.device = device
    
    def forward(self, input, binary_labels, labels, freeze: []):
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward(input, binary_labels, freeze[i])
        
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
        
        batch_input = None
        batch_labels = None

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
        validation2_dataloader = DataLoader(train, batch_size=len(train), num_workers=16, persistent_workers=True)
        validation_dataloader = DataLoader(validation, batch_size=len(validation), num_workers=16, persistent_workers=True)
        
        for epoch in tqdm(range(epochs)):

            for sample in train_dataloader:

                batch_input = torch.cat([sample['pos']['image'], sample['neg']['image']]).to(self.device)
                batch_binary_labels = torch.cat([sample['pos']['binary_label'], sample['neg']['binary_label']]).to(self.device)
                batch_labels = torch.cat([sample['pos']['label'], sample['neg']['label']]).to(self.device)

                self.forward(batch_input, batch_binary_labels, batch_labels, freeze)

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
            #print("myacc:", sum(correct==True)/len(labels))
            wandb_logging[f"{partition}_layer_{i}_acc"] = sum(correct==True)/len(labels)
            #wandb_logging[f"{partition}_layer_{i}_lr"] = layer.learning_rate
            
            layer.step_update(epoch)

        
        output = self.classification_layer.eval(input)
        self.classification_layer.step_update(epoch)
        
        predictions = torch.argmax(output, axis=1)
        wandb_logging[f"{partition}_prediction_acc"] = torch.sum((predictions==labels))/len(labels)

        wandb.log(wandb_logging, step=epoch)


    def test(self):
        pass