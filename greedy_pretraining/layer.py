import torch
import torch.nn as nn
import torch.optim as optim

import math

# Layer with backprop support
class ForwardLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, threshold_multiplier: float, optimizer: str = "SGD", learning_rate: float = 1e-3, weight_decay: float = 3e-4, momentum: float = 0.9, device="cuda"):
        super(ForwardLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        # TODO: Look at possible other init schemas
        self.linear = nn.Linear(input_size, output_size)

        #init as normal distribution as recommended
        torch.nn.init.normal_(
            self.linear.weight, mean=0, std=1 / math.sqrt(self.linear.weight.shape[0])
        )
        torch.nn.init.zeros_(self.linear.bias)

        self.activation_function = nn.ReLU()

        self.threshold_multiplier = threshold_multiplier
        self.threshold = output_size*threshold_multiplier

        self.ff_loss = nn.BCEWithLogitsLoss()
        
        if optimizer == "SGD":
            self.optimizer = optim.SGD(params=self.linear.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(params=self.linear.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            print("Invalid optimizer in Forward-Forward layer:", optimizer)
        


        self.running_means = torch.zeros(output_size, device=device) + 0.5
        self.mean_momentum = 0.9
        
        self.device = device
        self.to(device)
    
    def step_update(self, epoch):
        if epoch > 45:
            self.learning_rate_multiplier = 2*(1 + 60-epoch)/60 
            #lr * 2 * (1 + 10 - epoch) / 10
            self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * 2 * (1 + 60 - epoch) / 60
    
    def get_learning_rate(self):
        return self.learning_rate_multiplier*self.learning_rate
    

    def forward_forward(self, input, binary_labels, freeze=False):
        """
        Passes the input and labels through this layer and updates the weights according to the Forward-Forward algorithm

        Parameters
        ----------
        input: Tensor
            A 2-dimensional (batch_size, datapoint_size) tensor containing the datapoints that begin with a one-hot encoding of a label
        binary_labels: Tensor
            A 1-dimensional (batch_size) tensor where each entry specifies if the corresponding datapoint is correctly or incorrectly labeled
        freeze: bool, optional
            Prevents updating the layer weights if True (default is False)
            
        """

        # Normalize s.t. each number is on average of size 1
        # This is also important in order to not bring direct knowledge of positive/negative sample from previous layer
        z = input
        mean_a_squared = torch.mean(z ** 2, axis=1) ** 0.5
        z = z / (10e-10 + torch.tile(mean_a_squared, (self.input_size, 1)).T)
        z = z.detach() # dunno if necessary don't think it is
    
        self.optimizer.zero_grad()
        
        z = self.linear(z)
        z = self.activation_function(z)

        
        #z = self.norm(z) # Try batch norm instead?

        
        if not freeze:
            self.backward(z, binary_labels)
            
        return z.detach()
    
    def forward_backprop(self, input):
        z = self.linear(input)
        z = self.activation_function(z)
        return z

    def eval(self, input):
        with torch.no_grad(): # Probs useless function
            return self.forward_forward(input, None, freeze=True)
        
    def eval_backprop(self, input):
        with torch.no_grad(): # Probs useless function
            return self.forward_backprop(input, None, freeze=True)


    def backward(self, z, binary_labels):
        loss = self.layer_loss(z, binary_labels)
        loss += 0.03 * torch.mean(self.peer_norm(z))
        loss.backward()
        self.optimizer.step()

        
    def peer_norm(self, z):
        # TODO: Check how this changes the overall performance - seems to do nothing? Also Hinton version is non-square -> only because direct grad comp


        #self.running_means = self.mean_momentum * self.running_means.detach() + (1-self.mean_momentum)* torch.mean(z, dim=0) # div by batchsize???
        #return torch.mean(self.running_means) - self.running_means # sum or mean of this as well?
        
        #Only for positive samples according to other students
        mean_activity = torch.mean(z[:100], dim=0) #bsx2000 -> 2000
        #print("correct? peer loss", loss)
        # Why detach?
        self.running_means = self.running_means.detach() * self.mean_momentum + mean_activity * (
            1 - self.mean_momentum
        )
        

        peer_loss = (torch.mean(self.running_means) - self.running_means) ** 2

        return torch.mean(peer_loss)
        
        #scalar_outputs["Peer Normalization"] += peer_loss
        #scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

    def layer_loss(self, z, binary_labels):
        """
        Computes the layer loss as specified by the ForwardForward algorithm. Optimizes for the sum of squares to be greater than threshold if a positive sample is given and less than threshold if a negative sample is given
        """

        
        sum_squares = torch.sum(z**2, dim=1)

        logits = sum_squares - self.threshold

        loss = self.ff_loss(logits, binary_labels)
        return loss
