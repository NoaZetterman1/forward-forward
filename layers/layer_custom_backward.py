import torch
import torch.nn as nn
import torch.optim as optim



class ForwardLayerCustomBackward(nn.Module):
    # Backprop implemented by hand
    def __init__(self, input_size: int, output_size: int, threshold_multiplier: float, device):
        super(ForwardLayerCustomBackward, self).__init__()

        self.weightsgrad = torch.zeros((output_size,input_size)).to(device)
        self.biasesgrad = torch.zeros((output_size)).to(device)
        self.learning_rate = 0.01
        self.learning_rate_multiplier = 1

        self.input_size = input_size
        self.output_size = output_size

        # TODO: Look at possible other init schemas
        self.linear = nn.Linear(input_size, output_size)

        """ #init as normal distribution as recommended
        torch.nn.init.normal_(
            self.linear.weight, mean=0, std=1 / math.sqrt(self.linear.weight.shape[0])
        )
        torch.nn.init.zeros_(self.linear.bias)"""

        self.norm = nn.BatchNorm1d(output_size)
        #self.norm = nn.GroupNorm(10)
        #self.norm = nn.LayerNorm(output_size)
        self.activation_function = nn.ReLU()

        self.threshold_multiplier = threshold_multiplier
        self.threshold = output_size*threshold_multiplier

        self.running_means = torch.zeros(output_size, device=device) + 0.5
        self.mean_momentum = 0.9
        


        self.device = device
        self.to(device)


    # Rename labels to something that makes more sense
    def step_update(self, epoch):
        #print(self.optimizer.param_groups[0]["lr"])
        if epoch > 30:
            self.learning_rate_multiplier = (1 + 2 * 50)/60 
            #lr * 2 * (1 + 10 - epoch) / 10
            #self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]/10 # * 2 * (1 + 30 - epoch) / 30
    
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
        with torch.no_grad():
            mean_a_squared = torch.mean(input ** 2, axis=1) ** 0.5
            z = input / (10e-10 + torch.tile(mean_a_squared, (self.input_size, 1)).T)
        
            #print("sum squares:", torch.sum(z**2, dim=1)[0])
            
            self.normstates = z
            z = self.linear(z) # row 144
            
            #print("b4", z[0])
            
            z = self.activation_function(z) # row 145
            #print("after ReLU", z[0])

            #Test Peer norm VS batch norm?

            #Do peer norm here

            # Mean activity across batch
            
            # the detach means that the gradient because of previous batches is not backpropagated. only the current mean activity is backpropagated
            # running_mean * 0.9 + mean_activity * 0.1
            
            # 2000
            # 1 = mean activation across entire layer
            # 
            #z = self.norm(z) # Should use my activation func
            #z = self.peer_norm(z) # 146 not

            

            if not freeze:
                self.backward(z, binary_labels) # packprop: 147-156
            

            #Do peer normalization here NOT
            # 146 - done after because it is unaffected by backward.
            """numcomp = z.shape[1]

            mean_a_squared = torch.mean(z ** 2, axis=1) ** 0.5
            z = z / (10e-10 + torch.tile(mean_a_squared, (numcomp, 1)).T)"""
            # 146 end

            z = z.detach()

        
        return z
    
    def eval(self, input):
        return self.forward(input, None, freeze=True)


    def backward(self, z, binary_labels):
        # Done by hand translated from matlab

        sum_squares = torch.sum(z**2, dim=1) # 147 mid part
        #print("ss:", sum_squares)
        #print(self.linear.weight)
        #print(z)
        #print(self.threshold)
        # threshold = z.shape[1]*threshold_factor???
        logits = sum_squares - self.threshold # 147 inner part
        #print(logits.shape) # 200
        #print(logits)
        #print(labels)
        
        posprobs = 1/(1+torch.exp(-logits)) #147
        #print(posprobs)

        #print(posprobs.shape) # 200
        arr = torch.where(binary_labels == 0, -posprobs, 1 - posprobs)
        #print(arr)
        # Change 1-posprobs to -negprobs for the negative examples
        repeated_diff_array = arr.unsqueeze(1).expand(-1, 2000) # 148
        dCbydin = repeated_diff_array * z # 148

        #print("dcbydin shape:", dCbydin.shape) # bs*2000

        self.running_means = self.mean_momentum * self.running_means + (1-self.mean_momentum)* torch.mean(z, dim=0) # 150

        dCbydin = dCbydin + 0.03 * (torch.mean(self.running_means) - self.running_means) # 151, maybe mean across diff dimension
        #print(self.normstates.shape) # 200 x 784
        #print(self.normstates.device) # 200 x 784
        posdCbydweights = (self.normstates.T @ dCbydin).T
        #print(posdCbydweights.shape) # 784 * 2000
        posdCbydbiases = sum(dCbydin)

        # Epsilon = learning rate
        # Epsgain = some multiplier which is to be increased or decreased during training instead of using epsilon

        # wc is "weight cost" = 0.001 -> L2 norm???-ish
        wc = 0.001

        # delay is used for smoothing gradient over minibatches =0.9
        delay = 0.9

        self.weightsgrad = delay * self.weightsgrad + (1-delay)*(posdCbydweights)/len(z) # 214
        self.biasesgrad = delay*self.biasesgrad + (1-delay)*(posdCbydbiases) / len(z) # 215
        self.linear.bias += + self.learning_rate * self.biasesgrad # 216

        #print(self.weightsgrad.shape)
        #print(self.linear.weight.shape)
        self.linear.weight += self.learning_rate * (self.weightsgrad -  wc*self.linear.weight) # 217