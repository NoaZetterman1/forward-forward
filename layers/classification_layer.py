import torch
import torch.nn as nn
import torch.optim as optim


class ClassificationLayer(nn.Module):
    """
    ClassificationLayer is a PyTorch nn.Module representing a standard classification layer, which can be used for evaluation in a Forward-Forward network.
    """

    def __init__(self, input_size, output_size, device):
        """
        Initialize ClassificationLayer.

        Parameters
        ----------
        input_size: int
            The size of the input to the layer.
        output_size: int
            The size of the output from the layer.
        device: str 
            The device on which the layer is placed (i.e. "cpu" or "cuda")
        """
        super(ClassificationLayer, self).__init__()

        self.linear = nn.Linear(input_size, output_size, bias=False)

        self.input_size = input_size
        self.output_size = output_size

        self.threshold = output_size

        self.classification_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(params=self.linear.parameters(), lr=0.008, weight_decay=3e-3, momentum=0.9)
        
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
            
        if epoch >= 40:
            self.learning_rate_multiplier = (1 + 2 * 30)/60 
            #lr * 2 * (1 + 10 - epoch) / 10
            self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * (1 + 60 - epoch) / 60
            #self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]/10

    def forward(self, input, classification_labels, freeze=False):
        """
        Passes the input and labels through this classification layer and updates the weights.

        Parameters
        ----------
        input: torch.Tensor
            Input tensor for the forward pass.
        classification_labels: torch.Tensor:
            Tensor containing true labels of the data.
        freeze: (bool, optional)
            Prevents updating the layer weights if True (default is False).

        Returns
        -------
        torch.Tensor
            Output tensor from the layer, as logits
        """
        #Slighly worse with normalization
        #mean_a_squared = torch.mean(input ** 2, axis=1) ** 0.5
        #z = input / (10e-10 + torch.tile(mean_a_squared, (self.input_size, 1)).T)
        #z = z.detach() # dunno if necessary don't think it is
        self.optimizer.zero_grad()


        z = self.linear(input)


        if not freeze:
            loss = self.layer_loss(z, classification_labels)
            loss.backward()
            self.optimizer.step()
        
        z = z.detach()
        return z
    
    def eval(self, input):
        return self.linear(input)

    
    def layer_loss(self, z, classification_labels):
        loss = self.classification_loss(z, classification_labels)
        return loss
    
