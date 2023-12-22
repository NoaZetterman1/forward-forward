import torch
from torch.utils.data import DataLoader


from tqdm import tqdm

import wandb

class ForwardForwardMulticlassification():
    """
    ForwardForwardMulticlassification is a class for training and evaluating a neural network with multiple forward layers
    followed by a classification layer connected to all the forward layers. Also uses the same method as ForwardForwardNeutral to 
    train the classification layer.
    """

    def __init__(self, device):
        self.forward_layers = []
        self.classification_layer = None
        self.device = device
    
    def forward(self, sample, freeze: []):

        input = torch.cat([sample['pos']['image'], sample['neg']['image']]).to(self.device)
        targets = torch.cat([sample['pos']['binary_label'], sample['neg']['binary_label']]).to(self.device)
        labels = torch.cat([sample['pos']['label'], sample['neg']['label']]).to(self.device)
        
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward(input, targets, freeze[i])

        # Do classification with neutral labels
        input = sample['neutral']['image'].to(self.device)
        targets = sample['neutral']['binary_label'].to(self.device)
        labels = sample['neutral']['label'].to(self.device)
        inputs = torch.zeros((input.shape[0],6000)).to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward(input, targets, freeze=True)
            if i != 0:
                inputs[:, (i-1)*2000:(i)*2000] = input
        
        output = self.classification_layer.forward(inputs, labels)

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
        validation2_dataloader = DataLoader(train, batch_size=len(train))
        validation_dataloader = DataLoader(validation, batch_size=len(validation))
        for epoch in tqdm(range(epochs)):
            for sample in train_dataloader:

                
                                

                self.forward(sample, freeze)
            self.evaluate(validation_dataloader, epoch, "val")
            self.evaluate(validation2_dataloader, epoch, "train")


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
        targets = data['neutral']['binary_label'].to(self.device)
        labels = data['neutral']['label'].to(self.device)
        inputs = torch.zeros((input.shape[0],6000)).to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
            if i != 0:
                inputs[:, (i-1)*2000:(i)*2000] = input
        
        output = self.classification_layer.eval(inputs)
        self.classification_layer.step_update(epoch)
        
        predictions = torch.argmax(output, axis=1)
        wandb_logging[f"{partition}_prediction_acc"] = torch.sum((predictions==labels))/len(labels)

        wandb.log(wandb_logging, step=epoch)
        #print("Validation accuracy:", torch.sum((predictions==labels))/len(labels), " lastlayer:", sum(correct==True)/len(labels))