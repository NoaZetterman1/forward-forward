import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import itertools
import wandb

class ForwardForwardNeutral():
    # Trains the classification layer with neutral samples - samples that has label input label "1/10" in all places
    def __init__(self, device):

        self.forward_layers = []
        self.classification_layer = None
        self.device = device
    
    def forward(self, sample, freeze: []):
        input = torch.cat([sample['pos']['image'], sample['neg']['image']]).to(self.device)
        binary_labels = torch.cat([sample['pos']['binary_label'], sample['neg']['binary_label']]).to(self.device)
        labels = torch.cat([sample['pos']['label'], sample['neg']['label']]).to(self.device)
        # freeze is to be able to freeze early layers when training later, for testing.
        
        # Train F-F Layers
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward_forward(input, binary_labels, freeze[i])

        # Train classification layers
        input = sample['neutral']['image'].to(self.device)
        binary_labels = sample['neutral']['binary_label'].to(self.device)
        labels = sample['neutral']['label'].to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward_forward(input, binary_labels, freeze=True)
        
        output = self.classification_layer.forward_forward(input, labels)

        return output
    def forward_backprop(self, input, labels):
        # freeze is to be able to freeze early layers when training later, for testing.
        for i, layer in enumerate(self.forward_layers):
            input = layer.forward_backprop(input)
        
        # Train classification layer on only "normal" labels ran through network
        output, loss = self.classification_layer.forward_backprop(input, labels)


        return output, loss
    
    def add_layer(self, layer):
        self.forward_layers.append(layer)
    
    def add_classification_layer(self, layer):
        self.classification_layer = layer

    def update_ff_learning_rate(self, epoch):
        for layer in self.forward_layers:
            layer.step_update(epoch)
        self.classification_layer.step_update(epoch)

    def train_ff(self, train, validation, batch_size, epochs):
        """
        train - train dataset
        validation - validation dataset
        
        batch_size - batch size for training
        epochs - epochs for training (and evaluating)
        """
        freeze = [False] * (len(self.forward_layers)+1) # May want to freeze early layers after some training
        # Training loop, give data and loop over and train that data, lr schedulers and more?
        

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)

        # Compute entire batch at once when evaluating metrics on train data
        train_for_validation_dataloader = DataLoader(train, batch_size=len(train), num_workers=16, persistent_workers=True)
        validation_dataloader = DataLoader(validation, batch_size=len(validation), num_workers=16, persistent_workers=True)
        
        # Try both greedy and non-greedy layerwise pre-training?
        for epoch in tqdm(range(epochs)):
            for sample in train_dataloader:

                self.forward(sample, freeze)

            self.evaluate(validation_dataloader, epoch, "val")
            self.evaluate(train_for_validation_dataloader, epoch, "train")
            self.update_ff_learning_rate(epoch)


    def train_backprop(self, train, validation, batch_size, epochs, learning_rate=0.001, weight_decay=1e-3):
        """
        train - train dataset
        validation - validation dataset
        
        batch_size - batch size for training
        epochs - epochs for training (and evaluating)
        """
        
        params = list(self.classification_layer.linear.parameters())
        for layer in self.forward_layers:
            params += list(layer.linear.parameters())

        backprop_optimizer = optim.Adam(itertools.chain(params), lr=learning_rate, weight_decay=weight_decay)


        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
        validation2_dataloader = DataLoader(train, batch_size=len(train), num_workers=16, persistent_workers=True)
        validation_dataloader = DataLoader(validation, batch_size=len(validation), num_workers=16, persistent_workers=True)


        for epoch in tqdm(range(epochs)):
            for sample in train_dataloader:
                backprop_optimizer.zero_grad()
                batch_input = sample['neutral']['image'].to(self.device)
                batch_labels = sample['neutral']['label'].to(self.device)

                _, loss = self.forward_backprop(batch_input, batch_labels)
                loss.backward()
                backprop_optimizer.step()
            
            if epoch == epochs-5:
                backprop_optimizer.param_groups[0]["lr"] = backprop_optimizer.param_groups[0]["lr"] /10

            #self.evaluate_backprop(validation_dataloader, epoch+30, "val")
            #self.evaluate_backprop(validation2_dataloader, epoch+30, "train")
            self.evaluate_backprop(validation_dataloader, epoch, "test")
            self.evaluate_backprop(validation2_dataloader, epoch, "train")

    def train(self, train, validation, learning_rate, epochs=1):
        freeze = [False] * (len(self.forward_layers)+1) # May want to freeze early layers after some training
        # Training loop, give data and loop over and train that data, lr schedulers and more?
        
        batch_input = None
        batch_labels = None

        train_dataloader = DataLoader(train, batch_size=100, shuffle=True, num_workers=16, persistent_workers=True)
        validation2_dataloader = DataLoader(train, batch_size=len(train), num_workers=16, persistent_workers=True)
        validation_dataloader = DataLoader(validation, batch_size=len(validation), num_workers=16, persistent_workers=True)
        
        for epoch in tqdm(range(epochs)):

            for sample in train_dataloader:

                self.forward(sample, freeze)

            self.evaluate(validation_dataloader, epoch, "val")
            self.evaluate(validation2_dataloader, epoch, "train")

            if epoch == 50:
                freeze = [False] * (len(self.forward_layers)+1) # May want to freeze early layers after some training

            #if epoch==10:
            #    freeze[0] = True

    def evaluate(self, validation_dataloader, epoch, partition):
        data = next(iter(validation_dataloader))
        
        input = torch.cat([data['pos']['image'], data['neg']['image']]).to(self.device)
        binary_labels = torch.cat([data['pos']['binary_label'], data['neg']['binary_label']]).to(self.device)
        labels = torch.cat([data['pos']['label'], data['neg']['label']]).to(self.device)

        wandb_logging = {}
        input1=None
        #combined_score = 0
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
            input1 = input

            sum_squares = torch.sum(input1**2, dim=1)
            
                
            #print("acc according to themm:", ff_accuracy)
            
            correct = torch.where(binary_labels==0, sum_squares<=2000, sum_squares>2000)
            #combined_score += sum(correct==True)/len(labels)
            #print("myacc:", sum(correct==True)/len(labels))
            wandb_logging[f"{partition}_layer_{i}_acc"] = sum(correct==True)/len(labels)
            
            #layer.step_update(epoch)
            
        input = data['neutral']['image'].to(self.device)
        labels = data['neutral']['label'].to(self.device)
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
        
        output = self.classification_layer.eval(input)
        #self.classification_layer.step_update(epoch)
        
        predictions = torch.argmax(output, axis=1)
        wandb_logging[f"{partition}_prediction_acc"] = torch.sum((predictions==labels))/len(labels)
        #wandb_logging["combined_score"] = torch.sum((predictions==labels))/len(labels)*4 + combined_score

        # Do some combined metric like each layer loss + prediction loss and maximize everything over this?
        # Cool graphs
        wandb.log(wandb_logging, step=epoch)

    def evaluate_backprop(self, validation_dataloader, epoch, partition):
        data = next(iter(validation_dataloader))
        

        input = data["original"].to(self.device)
        labels = data['label'].to(self.device)
        #input = data["image"].to(self.device)
        #binary_labels = data["binary_label"].to(self.device) # temp
        #labels = data["label"].to(self.device)
        wandb_logging = {}
        input1=None

        
        for i, layer in enumerate(self.forward_layers):
            input = layer.eval(input)
            #print("acc according to themm:", ff_accuracy)
            
            #correct = torch.where(binary_labels==0, sum_squares<=2000, sum_squares>2000)
            #print("myacc:", sum(correct==True)/len(labels))
            #wandb_logging[f"{partition}_layer_{i}_acc"] = sum(correct==True)/len(labels)
            #wandb_logging[f"{partition}_layer_{i}_lr"] = layer.learning_rate
            
        
        output = self.classification_layer.eval(input)
        #self.classification_layer.step_update(epoch)
        
        predictions = torch.argmax(output, axis=1)
        wandb_logging[f"{partition}_acc"] = torch.sum((predictions==labels))/len(labels)
        

        wandb.log(wandb_logging, step=epoch)

