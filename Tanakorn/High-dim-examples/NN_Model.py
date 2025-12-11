import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(),seed=42):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.seed = seed
        torch.manual_seed(seed)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(self.activation)  
        self.init_weights()  

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Glorot (Xavier) initialization
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)  

    def forward(self, x): #x1, x2, x3):
        inputs = x #torch.cat([x1, x2, x3], axis=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
        
class TwoLayerResBlock(nn.Module):
    """
    Residual block consisting of:
    FC -> activation -> FC -> (add skip)
    """
    def __init__(self, width, activation=nn.Tanh()):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = activation

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.activation(self.fc1(x))
        z = self.fc2(z)
        return x + z     # skip connection (pure residual) 

class ResidualMLP_6Layers(nn.Module):
    def __init__(self, input_dim, width, output_dim, activation=nn.Tanh(), seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.activation = activation

        # Input layer
        self.fc_in = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.zeros_(self.fc_in.bias)

        # Three residual blocks, each with 2 layers → total 6 FC layers
        self.blocks = nn.ModuleList([
            TwoLayerResBlock(width, activation),
            TwoLayerResBlock(width, activation),
            TwoLayerResBlock(width, activation)
        ])

        # Final output layer
        self.fc_out = nn.Linear(width, output_dim)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        h = self.activation(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        return self.fc_out(h)   
        
class ResidualMLP_MBlocks(nn.Module):
    def __init__(self, input_dim, width, output_dim, M,
                 activation=nn.Tanh(), seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.activation = activation

        # Input projection layer
        self.fc_in = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.zeros_(self.fc_in.bias)

        # M residual blocks
        self.blocks = nn.ModuleList(
            [TwoLayerResBlock(width, activation) for _ in range(M)]
        )

        # Final output layer
        self.fc_out = nn.Linear(width, output_dim)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        h = self.activation(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        return self.fc_out(h)        