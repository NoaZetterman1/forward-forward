import torch
import torch.nn as nn
import torch.optim as optim

class RecurrentForwardLayer(nn.Module):
    """
    RecurrentForwardLayer is a PyTorch nn.Module representing a recurrent forward layer in a neural network architecture.

    To add recurrent connections to this layer, the set_recurrent_connection_from method can be used.  Note that the a standard
    RNN uses set_recurrent_connection_from(self), but this class is not restricted to only having recurrent connections to the layer itself.
    """


    def __init__(self, input_size: int, output_size: int, threshold_multiplier: float, optimizer, normalize_recurrent, activation_fn, device):
        """
        Initialize RecurrentForwardLayer.

        Parameters
        ----------
        input_size: int
            The size of the input to the layer.
        output_size: int
            The size of the output from the layer.
        threshold_multiplier: float
            Multiplier for determining the threshold value.
        optimizer: str
            The optimizer to be used for weight updates ("Adam" or "SGD").
        normalize_recurrent: bool
            Flag indicating whether to normalize recurrent connections.
        activation_fn: nn.Module
            Activation function for the layer.
        device:
            The device on which the layer is placed (CPU or GPU).
        """
        super(RecurrentForwardLayer, self).__init__()


        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)

        self.normalize_recurrent = normalize_recurrent
        #torch.nn.init.normal_(
        #    self.linear.weight, mean=0, std=1 / math.sqrt(self.linear.weight.shape[0])
        #)
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        self.activation_function = activation_fn()

        self.threshold_multiplier = threshold_multiplier
        self.threshold = output_size*threshold_multiplier

        self.ff_loss = nn.BCEWithLogitsLoss()

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.linear.parameters(), lr=0.001)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(params=self.linear.parameters(), lr=0.001, momentum=0.9)
        
        self.reccurent_connections = []

        self.running_means = torch.zeros(output_size, device=device) + 0.5
        self.mean_momentum = 0.9
        
        # Initialize the output as "neutral" in size, or as 0.
        self.output = torch.zeros(output_size, device=device) #+ threshold_multiplier


        self.device = device
        self.to(device)

    def set_layernr(self, nr):
        self.layer_nr = nr
    
    def set_recurrent_connection_from(self, layer):
        """
        Sets a recurrent connection from the given layer to this layer. It is possible to connect multiple layer outputs to the same input layer, 
        and to connect any two layers together (but note that connecting layer #X to layer #X+n, n>0 would not yield a recurrent structure)

        Parameters
        ----------
        layer: 
            A neural network layer object (Layer, RecurrentForwardLayer).
        """
        self.reccurent_connections.append(layer)
    
    def initialize_out_connection(self, batch_size):
        self.output = torch.zeros((batch_size, self.output_size), device=self.device) # + self.threshold_multiplier

    def normalize(self, z):
        """
        Normalizes the vector using the L2-Norm

        Parameters
        ----------
        input: Tensor
            A 1-dimensional tensor to normalize.
        """
        mean_a_squared = torch.mean(z ** 2, axis=1) ** 0.5
        return z / (10e-10 + torch.tile(mean_a_squared, (z.shape[-1], 1)).T)
    

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

        #Normalize and add recurrent connections
        with torch.no_grad():
            z = input
            
            z = self.normalize(z)
            #if torch.sum(z**2, axis=1)[0] == 0:
            #    print("0s in layer:", self.layer_nr, "meanasquared:", mean_a_squared)
                #print(input)
            #print(torch.sum(z**2, axis=1))
                #print(z)
            # Append recurrent connections
            for connection in self.reccurent_connections:
                if self.normalize_recurrent:
                    z = torch.cat((z, self.normalize(connection.output)), dim=-1)
                else:
                    z = torch.cat((z, connection.output), dim=-1)
            
            

        self.optimizer.zero_grad()
        
        z = self.linear(z)
        
        z = self.activation_function(z)
        
        
        if not freeze:
            self.backward(z, binary_labels)
        

        self.output = z

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
        mean_activity = torch.mean(z[:1], dim=0) #batch_size x 2000 -> 2000
        
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
        z: Tensor
            A 1-dimensional (output_size) tensor with the computed outputs for this layer
        binary_labels: Tensor
            A 1-dimensional (batch_size) tensor where each entry specifies if the corresponding datapoint is correctly or incorrectly labeled
        """
        
        sum_squares = torch.sum(z**2, dim=1)
        #sum_nonsquare = torch.sum(z, dim=1)

        logits = sum_squares - self.threshold
        #logits = sum_nonsquare - self.threshold

        loss = self.ff_loss(logits, binary_labels)
        return loss
