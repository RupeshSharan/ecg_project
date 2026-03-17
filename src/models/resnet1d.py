"""
1D ResNet for ECG Classification
==================================
Deep residual network with 1D convolutions adapted from ResNet-18.
Uses skip connections to enable deeper networks without degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Basic residual block with two Conv1D layers."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet for single-lead ECG classification."""

    def __init__(self, input_length=360, num_classes=5, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2,
                               padding=7, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock1D(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    model = ResNet1D(input_length=360, num_classes=5)
    x = torch.randn(4, 1, 360)
    out = model(x)
    print(f"ResNet1D output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters    : {total:,}")
