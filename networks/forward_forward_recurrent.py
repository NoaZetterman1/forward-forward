import torch
from torch.utils.data import DataLoader

from data_sampler import collate_text

from tqdm import tqdm

import wandb

class ForwardForwardRecurrent():
    """
    ForwardForwardRecurrent is a class for training and evaluating a neural network with multiple forward layers where each layer may 
    be recurrent and each datapoint is a sequence of word vectors and a binary label. 
    """

    def __init__(self, device):

        self.forward_layers = []
        self.classification_layer = None
        self.device = device
    
    def compute_forward_layers(self, inputs, binary_labels, freeze: []):
        for layer in self.forward_layers:
            layer.initialize_out_connection(batch_size=len(inputs))

        for index in range(inputs.shape[1]):
            input_slice = inputs[:,index,:]
            for i, layer in enumerate(self.forward_layers):
                input_slice = layer.forward(input_slice, binary_labels, freeze[i])
        
        return input_slice

    def eval_forward_layers(self, inputs):
        for layer in self.forward_layers:
            layer.initialize_out_connection(batch_size=len(inputs))

        for index in range(inputs.shape[1]):
            input_slice = inputs[:,index,:]

            for layer in self.forward_layers:
                input_slice = layer.eval(input_slice)
        
        return input_slice


    def forward(self, sample, freeze: []):
        word_vectors = sample['word_vectors'].to(self.device)

        label = sample['labels'].to(self.device)
        long_label = sample['labels'].type(torch.long)
        long_label = long_label.to(self.device)
        
        # Train F-F Layers
        last_layer_output = self.compute_forward_layers(word_vectors, label, freeze)
        
        
        output = self.classification_layer.forward(last_layer_output, long_label)

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

        train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_text, shuffle=False, num_workers=8, prefetch_factor=2, persistent_workers=True)

        validation_dataloader = DataLoader(validation, batch_size=1, collate_fn=collate_text, num_workers=8, shuffle=False, prefetch_factor=2, persistent_workers=True)
        
        for epoch in tqdm(range(epochs)):
            for sample in train_dataloader:
                self.forward(sample, freeze)
            
            #self.evaluate(train_dataloader, epoch=epoch, partition="train")

            self.evaluate(validation_dataloader, epoch=epoch, partition="val")

    def evaluate(self, dataloader, epoch, partition):
        wandb_logging = {}

        labels = []
        correct = torch.zeros(len(self.forward_layers)).to(self.device)
        #predictions_l1 = []
        #predictions_l0 = []

        for datapoint in dataloader:
            #print(datapoint["natural_sentence"])
            sentence = datapoint['word_vectors'].to(self.device)
            label = datapoint['labels'].to(self.device)
            

            for layer in self.forward_layers:
                layer.initialize_out_connection(batch_size=len(sentence))
            #predictions_l0.append([])
            #predictions_l1.append([])
            for index in range(sentence.shape[1]):
                # Get word at index for the sentence
                input_slice = sentence[:,index,:]

                for i, layer in enumerate(self.forward_layers):
                    input_slice = layer.eval(input_slice)
                    #if i == 0:
                    #    predictions_l0[-1].append(torch.sum(input_slice**2, dim=1).detach().cpu().numpy()[0]-layer.threshold)
                    #if i == 1:
                            #print(layer.threshold) # 30
                    #        predictions_l1[-1].append(torch.sum(input_slice**2, dim=1).detach().cpu().numpy()[0]-layer.threshold)

                    if index == sentence.shape[1]-1:
                        sum_squares = torch.sum(input_slice**2, dim=1)
                        
                        
                        
                            
                        correct[i] += sum(torch.where(label==0, sum_squares<layer.threshold, sum_squares>layer.threshold)==True)

            labels.append(label)
        
        labels = torch.stack(labels)
        
        #output = self.classification_layer.eval(forward_outputs)
        #for i in range(len(self.forward_layers)):
        #    wandb_logging[f"{partition}_layer_{i}_acc"] = correct[i]/(labels.numel())
        #predictions = torch.argmax(output, axis=2)
        #print("labels:     ", labels)
        #print("L0:", predictions_l0)
        #print("RecurrentLayer:", predictions_l1)
        #print("predictions:", predictions)
        #if torch.sum((correct[0]))/labels.numel() > 0.75:
        #print("LLPRED:", (torch.sum((correct[0]))/labels.numel()).cpu().numpy()*100)
        #print("acc:", torch.sum((predictions==labels))/len(labels))
        wandb_logging[f"{partition}_acc"] = (torch.sum((correct[0]))/labels.numel()).cpu().numpy()*100

        wandb.log(wandb_logging, step=epoch)


    def test(self, vectorized_sentence):
        output = self.eval_forward_layers(vectorized_sentence)

        result = self.classification_layer.eval(output)

        print("Predictions:", result)
