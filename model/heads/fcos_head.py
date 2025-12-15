import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class Scale(nn.Module):  # learning scale for each model
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class FCOSTargetGenerator:
    def __init__(self, obj_sizes_of_interest, center_sample=True, radius=1.5):
        self.obj_sizes_of_interest = obj_sizes_of_interest
        self.center_sample = center_sample
        self.radius = radius

    def __call__(self, locations, targets, fpn_strides):
        """
        Generate targets for all images in batch

        Args:
            locations: list of [H*W, 2] tensors (x, y coords) for each FPN level
            targets: list of dicts with 'boxes' [N, 4] and 'labels' [N] for each image
            fpn_strides: list of stride values [8, 16, 32, 64, 128]

        Returns:
            labels: [B, num_locations] class labels (0 = background)
            reg_targets: [B, num_locations, 4] bbox regression targets (l,t,r,b)
        """
        batch_size = len(targets)
        labels_all = []
        reg_targ_all = []
        
        for img_index in range(batch_size):
            gt_boxes = targets[img_index]['boxes']
            gt_labels = targets[img_index]['labels']
            
            labels_per_img = []
            reg_targ_per_img = []
            
            for level, (loc_per_level, stride) in enumerate(zip(locations, fpn_strides)):
                size_range = self.obj_sizes_of_interest[level]
                labels, reg_targ = self.compute_targets_single_level(loc_per_level, gt_boxes, gt_labels, size_range, stride)
                
                labels_per_img.append(labels)
                reg_targ_per_img.append(reg_targ)
            
            #concatenating grid points across FPN levels into one long list of points
            labels_all.append(torch.cat(labels_per_img, dim=0))
            reg_targ_all.append(torch.cat(reg_targ_per_img, dim=0))
        
        return torch.stack(labels_all), torch.stack(reg_targ_all)
            

class FCOSHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        prior_prob = getattr(cfg.MODEL.FCOS, "PRIOR_PROB", 0.01)
        num_levels = 5

        self.fpn_strides = [8, 16, 32, 64, 128]
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.CENTER_SAMPLE_RADIUS

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
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

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
            bbox_reg.append(torch.exp(self.scales[l](bbox_l)))
            centerness.append(self.centerness(bbox_feat))

        return logits, bbox_reg, centerness
