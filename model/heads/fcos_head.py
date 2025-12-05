import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """
    Args:
        cfg: config with
            cfg.MODEL.FCOS.NUM_CONVS
            cfg.MODEL.FCOS.NUM_CLASSES
            cfg.MODEL.FCOS.PRIOR_PROB (optional, default 0.01)
        in_channels: number of channels in FPN features (e.g. 256)
    """
    def __init__(self, cfg, in_channels):
        super().__init__()

        num_convs    = cfg.MODEL.FCOS.NUM_CONVS
        num_classes  = cfg.MODEL.FCOS.NUM_CLASSES
        prior_prob   = getattr(cfg.MODEL.FCOS, "PRIOR_PROB", 0.01)
        num_levels   = 5   

        # classification tower
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # regression tower
        bbox_tower = []
        for _ in range(num_convs):
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU(inplace=True))
        self.bbox_tower = nn.Sequential(*bbox_tower)

        # prediction heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred  = nn.Conv2d(in_channels, 4,           3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1,           3, padding=1)

        # per-level scales
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # weight initialization 
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.cls_logits,
            self.bbox_pred,
            self.centerness,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # focal-loss-friendly bias init for classification
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):
        """
        features: list of FPN maps [P3..P7], each [B, C, H, W]
        returns:
            logits:     list [B, num_classes, H, W]
            bbox_reg:   list [B, 4,           H, W]  (positive distances)
            centerness: list [B, 1,           H, W]
        """
        logits = []
        bbox_reg = []
        centerness = []

        for l, feature in enumerate(features):
            # classification branch
            cls_feat = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_feat))

            # regression + centerness branch
            bbox_feat = self.bbox_tower(feature)
            bbox_l = self.bbox_pred(bbox_feat)

            # Scale per level + exp to enforce positivity (exact FCOS behavior)
            bbox_reg.append(torch.exp(self.scales[l](bbox_l)))

            centerness.append(self.centerness(bbox_feat))

        return logits, bbox_reg, centerness
