import torch
from torch import nn, Tensor

class CNA(nn.Sequential): # Basic building block 
    def __init__(
        self,
        in_channels, 
        out_channels, 
        padding=None, 
        stride: int = 1, 
        kernel_size: int = 3, # Size of the convolutional kernel
        groups: int = 1, # Add groups parameter for group convolutions
        norm_layer = nn.BatchNorm2d,
        activation_layer = nn.ReLU, 
        bias = False
    ) -> None:
        if padding is None: 
            padding = (kernel_size - 1) // 2 # same padding formula
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias))
        layers.append(norm_layer(out_channels))
        if activation_layer is not None: 
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)

class Stem(CNA):
    """
    First layer of the network. 
    Reduces image size (stride=2) and increases channels.
    """
    def __init__(self, width_in, width_out) -> None:
        super().__init__(width_in, width_out, kernel_size=3, stride=2)

class BottleNeckTransform(nn.Sequential):
    """
    The main computation path inside a residual block:
    - 1x1 conv → shrink channels
    - 3x3 grouped conv → process features
    - 1x1 conv → expand channels back
    """
    def __init__(
        self, 
        width_in, 
        width_out, 
        stride, 
        group_width, 
        norm_layer=nn.BatchNorm2d, 
        bottleneck_multiplier = 1.0, 
        activation_layer=nn.ReLU
    ) -> None: 
        bottleneck_channels = int(width_out * bottleneck_multiplier)
        bottleneck_channels = int(round(bottleneck_channels/group_width)*group_width)
        super().__init__(
            # 1x1 compress
            nn.Conv2d(width_in, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            norm_layer(bottleneck_channels),
            activation_layer(inplace=True),
            
            # 3x3 grouped
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, 
                    padding=1, groups=bottleneck_channels//group_width, bias=False),
            norm_layer(bottleneck_channels),
            activation_layer(inplace=True),
            
            # 1x1 expand
            nn.Conv2d(bottleneck_channels, width_out, kernel_size=1, stride=1, bias=False),
            norm_layer(width_out)
        )
class ResidualBlock(nn.Module):
    """
    Adds skip connection:
        - main path = BottleNeckTransform
        - skip path = identity or 1x1 conv if shape changes
    Output = main + skip
    This helps training deep networks.
    """
    def __init__(
            self, 
            width_in, 
            width_out, 
            stride, 
            group_width,
            norm_layer, 
            activation_layer,
            bottleneck_multiplier = 1.0,
        ) -> None:
        super().__init__()
        self.f = BottleNeckTransform( # this is the main path
            width_in,
            width_out,
            stride,
            group_width,
            norm_layer=norm_layer,
            activation_layer=activation_layer,  # Fix: was norm_layer
            bottleneck_multiplier=bottleneck_multiplier,
        )
        # Skip connection
        self.down = None 
        if width_in != width_out or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(width_in, width_out, kernel_size=1, stride=stride, bias=False),
                norm_layer(width_out)
            )
        
        self.act = activation_layer(inplace=True)
        
    def forward(self, x) -> Tensor:
        identity = x
        out = self.f(x)
        if self.down is not None: 
            identity = self.down(x)
        out += identity
        return self.act(out)
    
class RegNetStage(nn.Sequential):
    """
    A stack of residual blocks:
        - first block may downsample (stride > 1)
        - rest keep size the same
    Each stage increases feature abstraction.
    """
    def __init__(
            self, 
            width_in, 
            width_out, 
            stride, 
            depth, 
            group_width, 
            norm_layer, 
            activation_layer, 
            bottleneck_multiplier,
            ) -> None:
        super().__init__()
        for i in range(depth):
            block = ResidualBlock(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                group_width,
                norm_layer,
                activation_layer,
                bottleneck_multiplier=bottleneck_multiplier,
            )
            self.add_module(f"block{i+1}", block)
    
class RegNetX800mfBackbone(nn.Module):
    def __init__(self, norm_layer = nn.BatchNorm2d, activation_layer = nn.ReLU) -> None:
        super().__init__()
        # Stem
        self.stem = Stem(3, 32) 
        # Stage parameters
        group_width = 16
        bottleneck_multiplier = 1.0
        
        self.stage1 = RegNetStage(32, 64, stride=2, depth=1, group_width=group_width, norm_layer=norm_layer, activation_layer=activation_layer, bottleneck_multiplier=bottleneck_multiplier)
        self.stage2 = RegNetStage(64, 128, stride=2, depth=3, group_width=group_width, norm_layer=norm_layer, activation_layer=activation_layer, bottleneck_multiplier=bottleneck_multiplier)
        self.stage3 = RegNetStage(128, 288, stride=2, depth=7, group_width=group_width, norm_layer=norm_layer, activation_layer=activation_layer, bottleneck_multiplier=bottleneck_multiplier)
        self.stage4 = RegNetStage(288, 672, stride=2, depth=5, group_width=group_width, norm_layer=norm_layer, activation_layer=activation_layer, bottleneck_multiplier=bottleneck_multiplier)
        
    def forward(self, x) -> Tensor:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

def main():
    """
    {
        "C2": low-level features,
        "C3": mid-level features,
        "C4": high-level features,
        "C5": very high-level features
    }
    """
    model = RegNetX800mfBackbone()
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    for name, fm in outputs.items():
        print(f"{name}: {fm.shape}")

if __name__ == "__main__":
    main()