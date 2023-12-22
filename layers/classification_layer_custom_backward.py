import torch
import torch.nn as nn
import torch.optim as optim


class ClassificationLayer(nn.Module):
    # Same as classification layer now, used for testing loss stufff
    def __init__(self, input_size, output_size, device):
        super(ClassificationLayer, self).__init__()

        self.linear = nn.Linear(input_size, output_size, bias=False) # Bias=False?

        self.threshold = output_size

        self.classification_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.linear.parameters(), lr=0.001)

        # Optimizer as implemented by other students
        #self.optimizer = torch.optim.SGD(params=self.linear.parameters(), lr=1e-2, weight_decay=3e-3, momentum=0.9)
        self.device = device
        self.to(device)

    def step_update(self, epoch):
        if epoch >= 30:
            self.learning_rate_multiplier = (1 + 2 * 30)/60 
            #lr * 2 * (1 + 10 - epoch) / 10
            #self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]/10 # * 2 * (1 + 30 - epoch) / 30
            #self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]/10

    def forward(self, input, classification_labels, freeze=False):
        input = input.detach()
        #Labels here actually refer to if it is a true or a false datapoint
        
        self.optimizer.zero_grad()
        #Layer norm for the inputs first? - Or at the end?
        z = input

        # print("Input shape:", z.shape)

        z = self.linear(z) # W*z + b

        #z = z - torch.max(z, dim=-1, keepdim=True)[0] #Does this even make a diff?
        # print("z shape:", z.shape)
        # print(z)

        #More normalization?

        # sum_z = torch.sum(z, dim=1)
        # print("sum z:", sum_z, " with threshold???", self.threshold)

        #TODO: Pick "real"/"fake" values - Maybe before sent in here (0=false, 1=True)

        

        # print("Layer loss:", loss)
        # print()
        if not freeze:
            loss = self.layer_loss(z, classification_labels)
            loss.backward()
            self.optimizer.step()
        #Do .backward() here when layer is evaluated // do it for each layer when iterating over the thing.

        
        return z.detach()
    
    def eval(self, input):
        return self.linear(input)

    
    def layer_loss(self, z, classification_labels):





        loss = self.classification_loss(z, classification_labels)
        return loss
    
