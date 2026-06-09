"""
U-Net with pre-trained ResNet34 encoder.

Encoder: torchvision ResNet34 (ImageNet pre-trained)
Decoder: U-Net-style with skip connections from ResNet layers

Skip connections:
    x0: after conv1+bn1+relu  → (64ch,  H/2)
    x1: after layer1           → (64ch,  H/4)
    x2: after layer2           → (128ch, H/8)
    x3: after layer3           → (256ch, H/16)
    x4: after layer4 (bottleneck) → (512ch, H/32)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """Upsample → Concat with skip → DoubleConv"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net with pre-trained ResNet34 encoder.

    Args:
        in_channels: input channels (default 3 for RGB)
        out_channels: output segmentation channels (default 1 for binary)
        pretrained: use ImageNet pre-trained weights (default True)
    """

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()

        # ── Load pre-trained ResNet34 ──
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)

        # Encoder layers
        self.enc0 = nn.Sequential(
            resnet.conv1,   # 7x7 conv, 3→64, stride 2
            resnet.bn1,
            resnet.relu,
        )  # out: 64ch, H/2
        self.pool0 = resnet.maxpool  # out: 64ch, H/4

        self.enc1 = resnet.layer1   # out: 64ch,  H/4
        self.enc2 = resnet.layer2   # out: 128ch, H/8
        self.enc3 = resnet.layer3   # out: 256ch, H/16
        self.enc4 = resnet.layer4   # out: 512ch, H/32

        # ── Decoder ──
        # Bottleneck processing
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec3 = DecoderBlock(512, 256, 256)   # 512→256 + skip 256 = 256
        self.dec2 = DecoderBlock(256, 128, 128)   # 256→128 + skip 128 = 128
        self.dec1 = DecoderBlock(128, 64, 64)     # 128→64  + skip 64  = 64
        self.dec0 = DecoderBlock(64, 64, 64)      # 64→64   + skip 64  = 64

        # Final upsample: 128×128 → 256×256 (no skip — original resolution lost in enc0 stride-2)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Output ──
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        # Handle non-3-channel input
        if in_channels != 3:
            old_conv = self.enc0[0]
            self.enc0[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Copy weights for first 3 channels, average for the rest
                with torch.no_grad():
                    self.enc0[0].weight[:, :3] = old_conv.weight
                    self.enc0[0].weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)       # (B, 64,  H/2,  W/2)
        x0p = self.pool0(x0)    # (B, 64,  H/4,  W/4)

        x1 = self.enc1(x0p)     # (B, 64,  H/4,  W/4)
        x2 = self.enc2(x1)      # (B, 128, H/8,  W/8)
        x3 = self.enc3(x2)      # (B, 256, H/16, W/16)
        x4 = self.enc4(x3)      # (B, 512, H/32, W/32)

        # Bridge
        x = self.bridge(x4)     # (B, 512, H/32, W/32)

        # Decoder with skip connections
        x = self.dec3(x, x3)    # (B, 256, H/16, W/16)
        x = self.dec2(x, x2)    # (B, 128, H/8,  W/8)
        x = self.dec1(x, x1)    # (B, 64,  H/4,  W/4)
        x = self.dec0(x, x0)    # (B, 64,  H/2,  W/2)
        x = self.final_up(x)    # (B, 32,  H,    W)
        x = self.final_conv(x)  # (B, 32,  H,    W)

        return self.out_conv(x)
