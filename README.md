# CN2VF-Net: Fire Detection System  

⚠️ **EDUCATIONAL PROJECT / PROOF-OF-CONCEPT** - See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for critical limitations

PyTorch implementation of CN2VF-Net (Convolutional Neural Network and Vision Transformer Framework) for fire detection.

## ⚠️ IMPORTANT LIMITATIONS

**Dataset Reality:**
- Trained on **80 ground-level fire images** (99% fire class, 0% smoke)
- Validated on **20 images only** (insufficient for robust evaluation)
- **NO smoke detection capability** (no smoke training data exists)
- **NOT tested on aerial/drone imagery** (despite architecture design)
- **100% validation accuracy** is due to tiny, homogeneous validation set

**This is a learning/demonstration project, NOT production-ready.**

## Architecture

- **Hybrid Backbone**: 3 CNN stages (MobileNetV3-style) + 2 Vision Transformer stages
- **Input**: 448×448×3 RGB images
- **Output**: 
  - Classification: Fire/Neutral (2 classes in practice, though 3 coded)
  - Bounding Box: Normalized [x, y, w, h] coordinates
- **Parameters**: ~1.26M (overengineered for current 80-image dataset)

## Dataset

**Kaggle Fire and Smoke Dataset**: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset

### Expected Dataset Structure

```
dataset_root/
├── images/
│   ├── train/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── val/
│       ├── image_001.jpg
│       └── ...
└── annotations/
    ├── train.json
    └── val.json
```

### Annotation Format

```json
[
  {
    "filename": "image_001.jpg",
    "class": "fire",
    "bbox": [x, y, w, h]
  },
  ...
]
```

## Installation

```bash
# Create virtual environment (already done in this workspace)
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision tqdm tensorboard pillow
```

## Usage

### 1. Download Dataset

Download the dataset from Kaggle and prepare it according to the structure above.

### 2. Test Dataset Loading

```bash
python dataset.py <path_to_dataset>
```

### 3. Train Model

```bash
python train.py \
  --data-root <path_to_dataset> \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --use-tensorboard \
  --output-dir ./output
```

#### Training Arguments

- `--data-root`: Path to dataset root directory (required)
- `--img-size`: Input image size (default: 448)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--cls-weight`: Classification loss weight (default: 1.0)
- `--bbox-weight`: Bounding box loss weight (default: 5.0)
- `--use-iou`: Use IoU loss for bbox instead of Smooth L1
- `--use-tensorboard`: Enable TensorBoard logging
- `--output-dir`: Output directory for checkpoints (default: ./output)

### 4. Monitor Training

If using TensorBoard:

```bash
tensorboard --logdir ./output/cn2vf_net_<timestamp>/tensorboard
```

### 5. Inference

```python
import torch
from cn2vf_net import CN2VFNet
from PIL import Image
import torchvision.transforms as T

# Load trained model
model = CN2VFNet(num_classes=3)
checkpoint = torch.load("output/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prepare image
transforms = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("test_image.jpg").convert("RGB")
image_tensor = transforms(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(image_tensor)
    
    # Get predictions
    class_probs = torch.softmax(output["cls_logits"], dim=1)
    predicted_class = torch.argmax(class_probs, dim=1).item()
    confidence = class_probs[0, predicted_class].item()
    bbox = output["bbox"][0]  # [x, y, w, h] normalized
    
    class_names = ["Fire", "Smoke", "Neutral"]
    print(f"Detected: {class_names[predicted_class]} ({confidence*100:.1f}%)")
    print(f"BBox: {bbox.tolist()}")
```

## Model Components

### Files

- `cn2vf_net.py` - Main model architecture
- `dataset.py` - Dataset loader for Kaggle Fire/Smoke dataset
- `train.py` - Training script with validation and checkpointing
- `test_cn2vf_net.py` - Comprehensive test suite (15 tests)

### Architecture Components

1. **ConvBNAct**: Lightweight Conv-BN-Activation block with GELU
2. **InvertedResidual**: MobileNetV3-style inverted residual blocks
3. **PatchEmbed**: Converts CNN feature maps to token sequences
4. **MHSABlock**: Multi-Head Self-Attention + FFN with LayerNorm
5. **TransformerStage**: Stacked Transformer blocks
6. **TokenDownsample**: Efficient spatial downsampling between ViT stages
7. **MultiScaleFusion**: Fuses CNN local features with ViT global features
8. **DetectionHead**: Lightweight detection head for classification + bbox

## Performance Notes

### Low-Latency Optimizations

- MobileNetV3-style inverted residuals (lightweight)
- Depthwise separable convolutions in fusion/head
- Shallow Transformer depths (2 blocks per stage)
- Efficient token downsampling
- Only 1.26M parameters

### Training Tips

1. **Start with pre-trained weights**: If available, initialize the CNN backbone with ImageNet pre-trained MobileNetV3 weights
2. **Learning rate**: Start with 1e-4, reduce by 10x if loss plateaus
3. **Batch size**: Increase if GPU memory allows (8-16 for 448×448 images)
4. **Data augmentation**: Already includes flip, color jitter; add rotation if needed
5. **Class imbalance**: If dataset is imbalanced, use weighted CrossEntropy
6. **Bounding box loss weight**: Default 5.0 works well, tune if bbox accuracy is poor

## Testing

Run the comprehensive test suite:

```bash
python test_cn2vf_net.py
```

Tests include:
- Model instantiation and forward pass
- Multiple batch sizes
- Gradient flow
- All component modules
- Output shape and range validation

**All 15 tests passed ✓**

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- Pillow
- tqdm
- tensorboard (optional, for training visualization)

## Citation

If you use this implementation, please cite:

```
CN2VF-Net: Convolutional Neural Network and Vision Transformer Framework
for Fire and Smoke Detection in High-Altitude Drone Imagery
```

## License

MIT License

## Dataset Source

Dataset: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
