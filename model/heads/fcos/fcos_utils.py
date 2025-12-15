import torch
import torch.nn as nn

class Scale(nn.Module):
    """
    A learnable scale parameter for FCOS.
    """
    def __init__(self, init_value = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value))) # init_value is needed because nn.Parameter does not accept int
        
    def forward(self, x):
        return x * self.scale
    
    