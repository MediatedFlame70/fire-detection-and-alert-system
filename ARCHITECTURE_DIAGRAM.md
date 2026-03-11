# CN2VF-Net Architecture Diagram

## Model Architecture Flow

```
INPUT IMAGE (448×448×3)
         |
         ▼
┌─────────────────────────────────────────┐
│         STEM LAYER                       │
│  Conv(3×3, stride=2) + BN + GELU        │
│  448×448×3 → 224×224×24                 │
└─────────────────────────────────────────┘
         |
         ▼
┌─────────────────────────────────────────┐
│      CNN STAGE 1 (3 blocks)             │
│  InvertedResidual × 3                   │
│  Expand Ratio: 4.0                      │
│  224×224×24 → 112×112×24 → 56×56×40     │
└─────────────────────────────────────────┘
         |
         ▼
┌─────────────────────────────────────────┐
│      CNN STAGE 2 (3 blocks)             │
│  InvertedResidual × 3                   │
│  Expand Ratio: 4.0                      │
│  56×56×40 → 28×28×48                    │
└─────────────────────────────────────────┘
         |
         ▼
┌─────────────────────────────────────────┐
│      CNN STAGE 3 (3 blocks)             │
│  InvertedResidual × 3                   │
│  Expand Ratio: 4.0                      │
│  28×28×48 → 28×28×80                    │
└─────────────────────────────────────────┘
         |
         ├──────────────────────────┐
         |                          |
         ▼                          |
┌─────────────────────────────────────────┐
│      PATCH EMBEDDING                     │
│  Conv(1×1) + Reshape                    │
│  28×28×80 → 784×128                     │
│  (H×W×C → N×D)                          │
└─────────────────────────────────────────┘
         |
         ▼
┌─────────────────────────────────────────┐
│  TRANSFORMER STAGE 1 (2 blocks)         │
│  Multi-Head Self-Attention              │
│  - Heads: 4                             │
│  - Dimension: 128                       │
│  - Depth: 2                             │
│  784 tokens × 128 dim                   │
└─────────────────────────────────────────┘
         |
         ▼
┌─────────────────────────────────────────┐
│      TOKEN DOWNSAMPLING                  │
│  Conv(3×3, stride=2) + Reshape          │
│  784×128 → 196×160                      │
│  (28×28 → 14×14)                        │
└─────────────────────────────────────────┘
         |
         ▼                          |
┌─────────────────────────────────────────┐
│  TRANSFORMER STAGE 2 (2 blocks)         │
│  Multi-Head Self-Attention              │
│  - Heads: 5                             │
│  - Dimension: 160                       │
│  - Depth: 2                             │
│  196 tokens × 160 dim                   │
└─────────────────────────────────────────┘
         |                          |
         └──────────┬───────────────┘
                    ▼
         ┌─────────────────────────┐
         │   MULTI-SCALE FUSION    │
         │  Concatenate + Conv     │
         │  - c2: 28×28×48         │
         │  - c3: 28×28×80         │
         │  - t2: 14×14×160→28×28  │
         │  → 28×28×224            │
         └─────────────────────────┘
                    |
                    ▼
         ┌─────────────────────────┐
         │    DETECTION HEAD       │
         │  Global Avg Pool        │
         │  + FC Layers            │
         │  ┌─────────────────┐    │
         │  │ Classification  │    │
         │  │   (3 classes)   │    │
         │  └─────────────────┘    │
         │  ┌─────────────────┐    │
         │  │  Bounding Box   │    │
         │  │  (4 coordinates)│    │
         │  └─────────────────┘    │
         └─────────────────────────┘
                    |
                    ▼
         ┌─────────────────────────┐
         │       OUTPUT            │
         │  - Class: Fire/Smoke/   │
         │           Neutral       │
         │  - BBox: [x, y, w, h]   │
         │  - Confidence Score     │
         └─────────────────────────┘
```

## Detailed Component Specifications

### 1. **InvertedResidual Block** (MobileNetV3-style)
```
Input (C_in)
    |
    ▼
[Expand Conv 1×1] → C_in × 4.0 (hidden_dim)
    |
    ▼
[BatchNorm + GELU]
    |
    ▼
[DepthWise Conv 3×3] → stride 1 or 2
    |
    ▼
[BatchNorm + GELU]
    |
    ▼
[Project Conv 1×1] → C_out
    |
    ▼
[BatchNorm]
    |
    ▼
[Residual Connection] (if stride=1 and C_in=C_out)
    |
    ▼
Output (C_out)
```

### 2. **Multi-Head Self-Attention Block**
```
Input Tokens (N × D)
         |
         ├─────────┬─────────┐
         ▼         ▼         ▼
       Query      Key      Value
         |         |         |
         └────┬────┴────┐    |
              ▼         ▼    |
           Attention = softmax(QK^T/√d)
                      |       |
                      └───┬───┘
                          ▼
                    Attention × V
                          |
                          ▼
                   [Linear Projection]
                          |
                          ▼
                   [Residual + LayerNorm]
                          |
                          ▼
                      [MLP Block]
                   (Linear → GELU → Linear)
                          |
                          ▼
                   [Residual + LayerNorm]
                          |
                          ▼
                   Output Tokens (N × D)
```

### 3. **Multi-Scale Fusion**
```
CNN Features (c2: 28×28×48)  ─┐
                              │
CNN Features (c3: 28×28×80)  ─┼─→ [Concatenate]
                              │      |
ViT Features (t2: 14×14×160) ─┘      ▼
         |                    [Upsample 2×]
         |                           |
         └───────────────────────────┤
                                     ▼
                             [Conv 1×1 Fusion]
                                     |
                                     ▼
                               28×28×224
                                     |
                                     ▼
                            [Global Avg Pool]
                                     |
                                     ▼
                                   1×224
```

## Model Statistics

```
┌────────────────────────────────────────────────────┐
│  PARAMETER COUNT BREAKDOWN                         │
├────────────────────────────────────────────────────┤
│  Stem Layer:                    ~1,000 params      │
│  CNN Stages (3 stages):         ~500,000 params    │
│  ViT Stages (2 stages):         ~600,000 params    │
│  Multi-Scale Fusion:            ~100,000 params    │
│  Detection Head:                ~63,000 params     │
├────────────────────────────────────────────────────┤
│  TOTAL:                         1,264,759 params   │
└────────────────────────────────────────────────────┘
```

## Feature Map Dimensions

```
Layer                    Output Shape          Parameters
─────────────────────────────────────────────────────────
Input                    [B, 3, 448, 448]      -
Stem                     [B, 24, 224, 224]     1,776
CNN Stage 1 (Block 1)    [B, 24, 112, 112]     6,336
CNN Stage 1 (Block 2)    [B, 24, 112, 112]     6,336
CNN Stage 1 (Block 3)    [B, 40, 56, 56]       10,240
CNN Stage 2 (Block 1)    [B, 48, 28, 28]       20,352
CNN Stage 2 (Block 2)    [B, 48, 28, 28]       24,768
CNN Stage 2 (Block 3)    [B, 48, 28, 28]       24,768
CNN Stage 3 (Block 1)    [B, 80, 28, 28]       41,280
CNN Stage 3 (Block 2)    [B, 80, 28, 28]       68,800
CNN Stage 3 (Block 3)    [B, 80, 28, 28]       68,800
Patch Embed              [B, 784, 128]         10,368
ViT Stage 1 (Block 1)    [B, 784, 128]         132,736
ViT Stage 1 (Block 2)    [B, 784, 128]         132,736
Token Downsample         [B, 196, 160]         185,120
ViT Stage 2 (Block 1)    [B, 196, 160]         205,120
ViT Stage 2 (Block 2)    [B, 196, 160]         205,120
Multi-Scale Fusion       [B, 224, 28, 28]      86,240
Detection Head (Class)   [B, 3]                50,691
Detection Head (BBox)    [B, 4]                50,688
─────────────────────────────────────────────────────────
TOTAL PARAMETERS                               1,264,759
```

## Key Architecture Features

1. **Hybrid Design**: Combines CNN's local feature extraction with Transformer's global attention
2. **Lightweight**: 1.26M parameters suitable for edge deployment
3. **Multi-Scale**: Fuses features from CNN stage 2, CNN stage 3, and ViT stage 2
4. **Efficient Attention**: Uses 4-5 attention heads with smaller token counts
5. **Mobile-Optimized**: MobileNetV3-style inverted residuals with expand ratio 4.0
6. **Fixed Input**: 448×448 resolution optimized for drone imagery
7. **Dual Output**: Simultaneous classification and bounding box regression
