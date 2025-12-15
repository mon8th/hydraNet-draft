# HydraNet - Multi-Head Object Detection Framework

A PyTorch-based object detection framework with a modular multi-head architecture. Named after the mythological Hydra, this framework supports multiple detection heads that can be trained and deployed simultaneously or independently.

## 📋 Overview

HydraNet is designed as a flexible detection framework with:
- **Backbone**: RegNetX-800MF for efficient feature extraction
- **Neck**: Feature Pyramid Network (FPN) for multi-scale features
- **Multiple Heads**: Pluggable detection heads (currently implementing FCOS, more to come)

## 🏗️ Architecture

```
Input Image → RegNetX-800MF → FPN → ┬─ FCOS Head → Predictions
              (C2-C5)         (P3-P7) ├─ [Future Head 2]
                                      ├─ [Future Head 3]
                                      └─ [More heads...]
```

### Multi-Head Design

The framework is built to support multiple detection paradigms:
- **FCOS**: Anchor-free, center-based detection 🚧 (Work in progress)
- **Additional Heads**: Planned for future implementation (RetinaNet, etc.)

### Components

- **RegNetX-800MF Backbone**: Efficient convolutional network outputting multi-scale features (C2, C3, C4, C5)
- **FPN Neck**: Converts backbone features to pyramid levels P3-P7 with 256 channels each
- **FCOS Head**: Predicts class scores, bounding boxes, and centerness for each pyramid level

## 📁 Project Structure

```
HydraNet-drafts/
├── configs/
│   └── hydraNet.yaml          # Model and training configuration
├── model/
│   ├── backbone/
│   │   ├── RegNetx800mf.py    # RegNetX backbone implementation
│   │   └── resnet_backbone.py # Alternative ResNet backbone
│   ├── neck/
│   │   └── fpn_for_RegNetx800mf.py  # FPN implementation
│   └── heads/
│       └── fcos/              # FCOS detection head
│           ├── fcos_head.py
│           ├── fcos_inference.py
│           ├── fcos_targets.py
│           └── fcos_utils.py
├── data/                      # Dataset directory
├── utils/                     # Utility functions
└── main_test.py              # Main detector model
```

## 🚀 Features

- **Multi-Head Architecture**: Modular design for multiple detection heads
- **Pluggable Components**: Easy to add new backbones, necks, and heads
- **Multi-scale Detection**: Handles objects of various sizes across pyramid levels
- **Flexible Configuration**: YAML-based configuration for different setups
- **🚧 Work in Progress**: Currently implementing FCOS head and core framework

### Object Size Assignment
- P3: 0-64 pixels
- P4: 64-128 pixels
- P5: 128-256 pixels
- P6: 256-512 pixels
- P7: >512 pixels

## ⚙️ Configuration

Key parameters in `configs/hydraNet.yaml`:

```yaml
MODEL:
  BACKBONE: regnetx_800mf
  NECK: FPN (256 channels)
  FCOS:
    NUM_CONVS: 4
    NUM_CLASSES: 80
    SCORE_THRESH: 0.05
    NMS_THRESH: 0.6
```

## 🔧 Usage

```python
from main_test import FCOSDetector
from types import SimpleNamespace

# Load configuration
cfg = SimpleNamespace()  # Or load from YAML

# Initialize model
model = FCOSDetector(cfg)

# Forward pass
outputs = model(images)
# Returns: cls_scores, bbox_preds, centerness
```

## 📊 Training Setup

- **Batch Size**: 8
- **Epochs**: 50
- **Learning Rate**: 0.01
- **Optimizer**: SGD with momentum (0.9)
- **Weight Decay**: 0.0001

### Loss Components
- Classification Loss (weight: 1.0)
- Regression Loss (weight: 1.0)
- Centerness Loss (weight: 1.0)

## 🎯 Inference

- **Pre-NMS Top-N**: 1000 detections
- **Post-NMS Top-N**: 100 final detections
- **Score Threshold**: 0.05
- **NMS Threshold**: 0.6

## 📝 Notes

⚠️ **This is an active learning project** - work in progress!

Exploring multi-head object detection architectures. The framework is designed to be modular and extensible, allowing for experimentation with different detection paradigms using a shared backbone and neck.

Currently configured for COCO dataset (80 classes) but can be adapted for custom datasets.

## 🎯 Roadmap

- [ ] Complete FCOS detection head implementation
- [ ] Training pipeline and loss functions
- [ ] Inference and post-processing
- [ ] Additional detection heads (YOLO, RetinaNet, etc.)
- [ ] Multi-head training pipeline
- [ ] Head-specific loss balancing
- [ ] Ensemble predictions from multiple heads

## 🔗 References

- FCOS: Fully Convolutional One-Stage Object Detection
- RegNet: Designing Network Design Spaces
- Feature Pyramid Networks for Object Detection
