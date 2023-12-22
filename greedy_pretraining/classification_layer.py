import torch
import torch.nn as nn
import torch.optim as optim


class ClassificationLayer(nn.Module):


    def __init__(self, input_size, output_size, optimizer="Adam", learning_rate=0.001, weight_decay=3e-3, momentum=0.9, device="cuda"):
        super(ClassificationLayer, self).__init__()

        self.linear = nn.Linear(input_size, output_size, bias=False) # Bias=False? Why

        self.input_size = input_size
        self.output_size = output_size

        self.threshold = output_size

        self.classification_loss = nn.CrossEntropyLoss()

        #TODO: Select appropriate optimizer and LR depending on learning scheme
        if optimizer == "SGD":
            self.optimizer = optim.SGD(params=self.linear.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(params=self.linear.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            print("Invalid optimizer in Forward-Forward layer:", optimizer)
        #self.optimizer = optim.SGD(params=self.linear.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        
        self.device = device
        self.to(device)

    def step_update(self, epoch):
        if epoch > 50:
            self.learning_rate_multiplier = 2*(1 + 60-epoch)/60 
            #lr * 2 * (1 + 10 - epoch) / 10
            self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * 2 * (1 + 60 - epoch) / 60
            #self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]/10


    def forward_forward(self, input, classification_labels, freeze=False):
        #Labels here actually refer to if it is a true or a false datapoint

        #Slighly worse with normalization
        #mean_a_squared = torch.mean(input ** 2, axis=1) ** 0.5
        #z = input / (10e-10 + torch.tile(mean_a_squared, (self.input_size, 1)).T)
        #z = z.detach()


        self.optimizer.zero_grad()
        #Layer norm for the inputs first? Or not?


        z = self.linear(input) # W*z + b


        if not freeze:
            loss = self.layer_loss(z, classification_labels)
            loss.backward()
            self.optimizer.step()
        
        z = z.detach()
        return z
    
    def forward_backprop(self, input, classification_labels, freeze=False):
        z = self.linear(input)
        loss = self.layer_loss(z, classification_labels)
        return z, loss
    
    def eval(self, input):
        return self.linear(input)

    
    def layer_loss(self, z, classification_labels):
        loss = self.classification_loss(z, classification_labels)
        return loss
    
