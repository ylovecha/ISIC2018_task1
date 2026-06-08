"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Reference: Ronneberger O, Fischer P, Brox T. MICCAI 2015.

Encoder path:   3 → 64 → 128 → 256 → 512
Bottleneck:     512 → 1024
Decoder path:   1024 → 512 → 256 → 128 → 64
Output:         64 → 1 (binary segmentation map)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d → BatchNorm → ReLU) × 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net with symmetric encoder-decoder architecture and skip connections.

    Args:
        in_channels (int): Number of input image channels. Default 3 (RGB).
        out_channels (int): Number of output segmentation channels. Default 1 (binary).
        features (list[int]): Encoder feature channel sizes.
            Default [64, 128, 256, 512]. Bottleneck is 2× the last encoder channel.
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Encoder (down-sampling path) ──
        prev_ch = in_channels
        for feat_ch in features:
            self.encoder.append(DoubleConv(prev_ch, feat_ch))
            prev_ch = feat_ch

        # ── Bottleneck ──
        bottleneck_ch = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_ch)

        # ── Decoder (up-sampling path) ──
        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feat_ch in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(feat_ch * 2, feat_ch, kernel_size=2, stride=2)
            )
            # After concatenation with skip connection, input channels double
            self.decoder.append(DoubleConv(feat_ch * 2, feat_ch))

        # ── Output layer ──
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: store skip-connection outputs
        skip_connections = []
        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # reverse to match decoder order
        for idx in range(len(self.decoder)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]

            # Handle size mismatch from odd input dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)  # skip connection
            x = self.decoder[idx](x)

        return self.out_conv(x)
