from typing import Callable, Optional

import torch
from torch import nn, Tensor

__all__ = ["RegNetX"]

# Con2d + BatchNorm + Activation
class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: 3,
        stride: 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        bias: bool = False
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias
            )
        )
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        
        super().__init__(*layers)

class SimpleStemIN(Conv2dNormActivation):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

class BottleneckTransform(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,  # It controls how many channels go into each group in the 3×3 convolution.
        bottleneck_multiplier: float, # It controls the width of the bottleneck layer.
    ) -> None:
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))  # inner width
        g = max(1, w_b // group_width)                       # number of groups
        
        self.add_module("a", Conv2dNormActivation(
            width_in, w_b, kernel_size=1, stride=1,
            norm_layer=norm_layer, activation_layer=activation_layer
        ))
        self.add_module("b", Conv2dNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g,
            norm_layer=norm_layer, activation_layer=activation_layer
        ))
        # No activation after "c"
        self.add_module("c", Conv2dNormActivation(
            w_b, width_out, kernel_size=1, stride=1,
            norm_layer=norm_layer, activation_layer=None
        ))

    def forward(self, x: Tensor) -> Tensor:
        a = self.a(x)
        b = self.b(a)
        c = self.c(b)
        return c
        
class ResBottleneckBlock(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
    ) -> None:
        super().__init__()

        # Bottleneck (1x1 → 3x3(group) → 1x1)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
        )

        # Projection for skip connection if needed
        self.down = None
        if stride != 1 or width_in != width_out:
            self.down = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None, 
            )

        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.f(x)

        if self.down is not None:
            identity = self.down(x)

        out = out + identity
        return self.activation(out)

    
class RegnetStage(nn.Sequential): # A stage is a sequence of blocks that operate at the same feature map resolution
    def __init__(self, width_in: int, width_out: int, stride: int, depth: int,
                 norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int,
                 bottleneck_multiplier: float, stage_index: int = 0
                 ) -> None:
        super().__init__()
        for i in range(depth):
            block = ResBottleneckBlock(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1, 
                norm_layer, 
                activation_layer,
                group_width,
                bottleneck_multiplier
            )
            self.add_module(f"block{stage_index + 1}_{i + 1}", block)
            
class  RegNetX800mfBackbone(nn.Module):
    def __init__(
        self,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        
        # stem 
        self.stem = SimpleStemIN(
            width_in=3,
            width_out=32,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        
        # stages
        group_width = 16
        bottleneck_multiplier = 1.0
        
        # out: 64 channels, /4 resolution
        self.stage1 = RegnetStage(
            width_in = 32,
            width_out= 64,
            stride = 2,
            depth = 1,
            norm_layer = norm_layer,
            activation_layer = activation_layer,
            group_width = group_width,
            bottleneck_multiplier = bottleneck_multiplier,
            stage_index = 1
        )
        
        # out: 128 channels, /8 resolution
        self.stage2 = RegnetStage(
            width_in = 64,
            width_out = 128,
            stride = 2,
            depth = 3,
            norm_layer = norm_layer,
            activation_layer = activation_layer,
            group_width = group_width,
            bottleneck_multiplier = bottleneck_multiplier,
            stage_index = 2
        )
        
        # out: 256 channels, /16 resolution
        self.stage3 = RegnetStage(
            width_in = 128,
            width_out = 288,
            stride = 2,
            depth = 7,
            norm_layer = norm_layer,
            activation_layer = activation_layer,
            group_width = group_width,
            bottleneck_multiplier = bottleneck_multiplier,
            stage_index = 3
        )
        
        # out: 512 channels, /32 resolution
        self.stage4 = RegnetStage(
            width_in = 288,
            width_out = 672,
            stride = 2,
            depth = 5,
            norm_layer = norm_layer,
            activation_layer = activation_layer,
            group_width = group_width,
            bottleneck_multiplier = bottleneck_multiplier,
            stage_index = 4
        )
        
    def forward(self, x):
        x = self.stem(x)   # /2
        c2 = self.stage1(x) # /4,  64 ch
        c3 = self.stage2(c2) # /8,  128 ch
        c4 = self.stage3(c3) # /16, 256 ch
        c5 = self.stage4(c4) # /32, 512 ch
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}
    
def main():
    model = RegNetX800mfBackbone()
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    for name, fm in outputs.items():
        print(f"{name}: {fm.shape}")
if __name__ == "__main__":
    main()