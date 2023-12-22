

import torch
import torch.nn as nn
import torch.optim as optim

import math

class ForwardLayer(nn.Module):
    """
    ForwardLayer is a PyTorch nn.Module representing a single layer in the Forward-Forward neural network architecture.
    """

    def __init__(self, input_size: int, output_size: int, threshold_multiplier: float, device):
        """
        Initialize ForwardLayer.

        Parameters:
        input_size: int
            The size of the input to the layer.
        output_size: int
            The size of the output from the layer.
        threshold_multiplier: float
            Multiplier for the threshold used in the Forward-Forward algorithm.
        device: str 
            The device on which the layer is placed (i.e. "cpu" or "cuda")
        """

        super(ForwardLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.linear = nn.Linear(input_size, output_size)

        torch.nn.init.normal_(
            self.linear.weight, mean=0, std=1 / math.sqrt(self.linear.weight.shape[0])
        )
        torch.nn.init.zeros_(self.linear.bias)

        #self.norm = nn.BatchNorm1d(output_size)
        #self.norm = nn.GroupNorm(10)
        #self.norm = nn.LayerNorm(output_size)
        self.activation_function = nn.ReLU()

        self.threshold_multiplier = threshold_multiplier
        self.threshold = output_size*threshold_multiplier

        self.ff_loss = nn.BCEWithLogitsLoss()

        #self.optimizer = optim.Adam(self.linear.parameters(), lr=0.0001)
        self.optimizer = optim.SGD(params=self.linear.parameters(), lr=1e-4*5, weight_decay=3e-4, momentum=0.9)
        


        self.running_means = torch.zeros(output_size, device=device) + 0.5
        self.mean_momentum = 0.9
        


        self.device = device
        self.to(device)


    def step_update(self, epoch):
        """
        Update the learning rate of the optimizer based on the epoch.

        Parameters
        ----------
        epoch: int
            Current epoch number.
        """
        if epoch > 30:
            self.learning_rate_multiplier = 2*(1 + 60-epoch)/60
            self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * 2 * (1 + 60 - epoch) / 60
    
    def get_learning_rate(self):
        return self.learning_rate_multiplier*self.learning_rate
    
    def forward(self, input, binary_labels, freeze=False):
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
        mean_a_squared = torch.mean(input ** 2, axis=1) ** 0.5
        z = input / (10e-10 + torch.tile(mean_a_squared, (self.input_size, 1)).T)
        z = z.detach()
        
        self.optimizer.zero_grad()
        
        z = self.linear(z)
        z = self.activation_function(z)

        
        #z = self.norm(z) # Try batch norm instead?

        
        if not freeze:
            self.backward(z, binary_labels)
            
        # Forget computations in autograd when leaving layer.
        return z.detach()
    
    def eval(self, input):
        with torch.no_grad():
            return self.forward(input, None, freeze=True)


    def backward(self, z, binary_labels):
        loss = self.layer_loss(z, binary_labels)
        loss += 0.03 * torch.mean(self.peer_norm(z))
        loss.backward()
        self.optimizer.step()

        
    def peer_norm(self, z):
        """
        Peer Normalization function

        Parameters
        ----------
        z: torch.Tensor 
            Input tensor of shape (bs, output_size), where bs is the batch size.

        Returns
        -------
        torch.Tensor
            Peer normalization loss of z.
        """

        mean_activity = torch.mean(z, dim=0) #bs x output_size -> output_size
        
        self.running_means = self.running_means.detach() * self.mean_momentum + mean_activity * (
            1 - self.mean_momentum
        )

        peer_loss = (torch.mean(self.running_means) - self.running_means) ** 2

        return torch.mean(peer_loss)

    def layer_loss(self, z, binary_labels):
        """
        Computes the layer loss as specified by the ForwardForward algorithm. Optimizes for the sum of squares to be greater than threshold if a positive sample is given and less than threshold if a negative sample is given
        
        Parameters
        ----------
        z: torch.Tensor
            Input tensor of shape (batch_size, output_size), representing the layer activations.
        binary_labels: torch.Tensor
            Binary labels indicating whether each sample is positive (1) or negative (0).

        Returns
        -------
        torch.Tensor
            Layer loss computed based on the ForwardForward algorithm.
        """

        
        sum_squares = torch.sum(z**2, dim=1)

        logits = sum_squares - self.threshold

        loss = self.ff_loss(logits, binary_labels)
        return loss
