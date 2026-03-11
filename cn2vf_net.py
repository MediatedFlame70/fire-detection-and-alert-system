import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Lightweight Conv-BN-Activation block."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, act_layer=nn.GELU):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer(),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    MobileNetV3-style inverted residual block (simplified for low latency).
    """

    def __init__(self, in_ch, out_ch, stride, expand_ratio=4.0, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if hidden_dim != in_ch:
            layers.append(ConvBNAct(in_ch, hidden_dim, kernel_size=1, stride=1, act_layer=act_layer))

        # Depthwise convolution
        layers.append(
            ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, act_layer=act_layer)
        )

        # Project back to output channels
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = x + out
        return out


class PatchEmbed(nn.Module):
    """
    Converts 2D CNN feature maps to 1D token sequences for Transformer stages.
    """

    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)
        # x: [B, embed_dim, H, W]
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        # x: [B, N, C] where N = H * W
        x = self.norm(x)
        return x, h, w


class MHSABlock(nn.Module):
    """
    Transformer block with MHSA + FFN + LayerNorm.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: [B, N, C]
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        # x: [B, N, C]

        ffn_input = self.norm2(x)
        x = x + self.ffn(ffn_input)
        # x: [B, N, C]
        return x


class TransformerStage(nn.Module):
    """
    A stack of Transformer blocks representing one ViT stage.
    """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MHSABlock(dim, num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)]
        )

    def forward(self, x):
        # x: [B, N, C]
        for blk in self.blocks:
            x = blk(x)
        # x: [B, N, C]
        return x


class TokenDownsample(nn.Module):
    """
    Efficient 2x spatial token downsampling between Transformer stages.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, h, w):
        # x: [B, N, C], N = H * W
        b, n, c = x.shape
        assert n == h * w, "Token count must match H*W"

        x2d = x.transpose(1, 2).reshape(b, c, h, w)
        # x2d: [B, C, H, W]
        x2d = F.avg_pool2d(x2d, kernel_size=2, stride=2)
        # x2d: [B, C, H/2, W/2]

        h2, w2 = x2d.shape[2], x2d.shape[3]
        x_ds = x2d.flatten(2).transpose(1, 2)
        # x_ds: [B, (H/2)*(W/2), C]
        x_ds = self.proj(x_ds)
        # x_ds: [B, (H/2)*(W/2), out_dim]
        return x_ds, h2, w2


class MultiScaleFusion(nn.Module):
    """
    Fuses high-resolution CNN local features with Transformer global semantics.
    """

    def __init__(self, c2_dim, c3_dim, t_dim, fuse_dim=128):
        super().__init__()
        self.c2_proj = ConvBNAct(c2_dim, fuse_dim, kernel_size=1, stride=1)
        self.c3_proj = ConvBNAct(c3_dim, fuse_dim, kernel_size=1, stride=1)
        self.t_proj = ConvBNAct(t_dim, fuse_dim, kernel_size=1, stride=1)

        # Depthwise separable fusion for lower latency
        self.fuse = nn.Sequential(
            ConvBNAct(fuse_dim * 3, fuse_dim * 3, kernel_size=3, stride=1, groups=fuse_dim * 3),
            ConvBNAct(fuse_dim * 3, fuse_dim, kernel_size=1, stride=1),
        )

    def forward(self, c2, c3, t2_map):
        # c2: [B, C2, 56, 56]
        # c3: [B, C3, 28, 28]
        # t2_map: [B, Ct, 14, 14]

        c2_p = self.c2_proj(c2)
        # c2_p: [B, F, 56, 56]

        c3_p = self.c3_proj(c3)
        c3_p = F.interpolate(c3_p, size=c2_p.shape[-2:], mode="bilinear", align_corners=False)
        # c3_p: [B, F, 56, 56]

        t_p = self.t_proj(t2_map)
        t_p = F.interpolate(t_p, size=c2_p.shape[-2:], mode="bilinear", align_corners=False)
        # t_p: [B, F, 56, 56]

        fused = torch.cat([c2_p, c3_p, t_p], dim=1)
        # fused: [B, 3F, 56, 56]
        fused = self.fuse(fused)
        # fused: [B, F, 56, 56]
        return fused


class DetectionHead(nn.Module):
    """
    Lightweight detection head for global class + single bounding box prediction.
    """

    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.pre = nn.Sequential(
            ConvBNAct(in_dim, in_dim, kernel_size=3, stride=1, groups=in_dim),
            ConvBNAct(in_dim, in_dim, kernel_size=1, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(in_dim, num_classes)
        self.box = nn.Linear(in_dim, 4)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.pre(x)
        # x: [B, C, H, W]
        x = self.pool(x).flatten(1)
        # x: [B, C]

        cls_logits = self.cls(x)
        # cls_logits: [B, 3] -> Fire/Smoke/Neutral

        bbox = torch.sigmoid(self.box(x))
        # bbox: [B, 4] normalized [x, y, w, h] in [0, 1]

        return cls_logits, bbox


class CN2VFNet(nn.Module):
    """
    CN2VF-Net: CNN + Vision Transformer hybrid for fire/smoke detection in drone imagery.

    Input:
        - image: [B, 3, 448, 448]

    Backbone:
        - CNN stage 1, 2, 3 (MobileNetV3-style inverted residuals)
        - ViT stage 1, 2 (MHSA + FFN)

    Output:
        - cls_logits: [B, 3]
        - bbox: [B, 4] as [x, y, w, h]
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # Stem: [B, 3, 448, 448] -> [B, 16, 224, 224]
        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)

        # CNN Stage 1: [B, 16, 224, 224] -> [B, 24, 112, 112]
        self.cnn_stage1 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=4.0),
            InvertedResidual(24, 24, stride=1, expand_ratio=4.0),
        )

        # CNN Stage 2: [B, 24, 112, 112] -> [B, 40, 56, 56]
        self.cnn_stage2 = nn.Sequential(
            InvertedResidual(24, 40, stride=2, expand_ratio=4.0),
            InvertedResidual(40, 40, stride=1, expand_ratio=4.0),
        )

        # CNN Stage 3: [B, 40, 56, 56] -> [B, 80, 28, 28]
        self.cnn_stage3 = nn.Sequential(
            InvertedResidual(40, 80, stride=2, expand_ratio=4.0),
            InvertedResidual(80, 80, stride=1, expand_ratio=4.0),
        )

        # Patch embedding from final CNN feature map: [B, 80, 28, 28] -> [B, 784, 128]
        self.patch_embed = PatchEmbed(in_ch=80, embed_dim=128)

        # ViT Stage 1: tokens remain at 28x28 grid, dim=128
        self.vit_stage1 = TransformerStage(dim=128, depth=2, num_heads=4, mlp_ratio=4.0, drop=0.0)

        # Token downsample: 28x28 -> 14x14, dim 128 -> 160
        self.token_down = TokenDownsample(in_dim=128, out_dim=160)

        # ViT Stage 2: tokens at 14x14 grid, dim=160
        self.vit_stage2 = TransformerStage(dim=160, depth=2, num_heads=5, mlp_ratio=4.0, drop=0.0)

        # Multi-scale fusion of CNN local features + Transformer global features
        self.fusion = MultiScaleFusion(c2_dim=40, c3_dim=80, t_dim=160, fuse_dim=128)

        # Lightweight detection head
        self.head = DetectionHead(in_dim=128, num_classes=num_classes)

    def forward(self, x):
        # x: [B, 3, 448, 448]

        x = self.stem(x)
        # x: [B, 16, 224, 224]

        c1 = self.cnn_stage1(x)
        # c1: [B, 24, 112, 112]

        c2 = self.cnn_stage2(c1)
        # c2: [B, 40, 56, 56]

        c3 = self.cnn_stage3(c2)
        # c3: [B, 80, 28, 28]

        tokens, h1, w1 = self.patch_embed(c3)
        # tokens: [B, 784, 128], h1=28, w1=28

        t1 = self.vit_stage1(tokens)
        # t1: [B, 784, 128]

        t2_tokens, h2, w2 = self.token_down(t1, h1, w1)
        # t2_tokens: [B, 196, 160], h2=14, w2=14

        t2_tokens = self.vit_stage2(t2_tokens)
        # t2_tokens: [B, 196, 160]

        b, n, c = t2_tokens.shape
        t2_map = t2_tokens.transpose(1, 2).reshape(b, c, h2, w2)
        # t2_map: [B, 160, 14, 14]

        fused = self.fusion(c2, c3, t2_map)
        # fused: [B, 128, 56, 56]

        cls_logits, bbox = self.head(fused)
        # cls_logits: [B, 3], bbox: [B, 4]

        return {
            "cls_logits": cls_logits,
            "bbox": bbox,
        }


if __name__ == "__main__":
    # Quick shape sanity check
    model = CN2VFNet(num_classes=3)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(2, 3, 448, 448)
        out = model(dummy)
        print("cls_logits shape:", out["cls_logits"].shape)  # [2, 3]
        print("bbox shape:", out["bbox"].shape)  # [2, 4]
