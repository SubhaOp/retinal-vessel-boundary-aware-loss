
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            raise ValueError("This implementation expects RGB input.")

        # ResNet34 encoder
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ImageNet normalization for pretrained encoder
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Decoder
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.dec0 = ConvBlock(64 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Normalize RGB input
        x = (x - self.mean) / self.std

        # Encoder
        stem = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(stem)

        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder with skip connections
        x = F.interpolate(
            e4, size=e3.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = F.interpolate(
            x, size=e2.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = F.interpolate(
            x, size=e1.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec1(torch.cat([x, e1], dim=1))

        x = F.interpolate(
            x, size=stem.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = self.dec0(torch.cat([x, stem], dim=1))

        x = F.interpolate(
            x, size=input_size,
            mode="bilinear", align_corners=False
        )

        # Return logits, not sigmoid probabilities
        return self.final(x)
