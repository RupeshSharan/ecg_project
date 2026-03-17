"""
1D Convolutional Neural Network for ECG Classification
=======================================================
3-block Conv1D → BatchNorm → ReLU → MaxPool architecture
followed by fully-connected classification head.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Lightweight 1D-CNN for single-lead ECG beat classification."""

    def __init__(self, input_length=360, num_classes=5, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # Compute the flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            feat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = CNN1D(input_length=360, num_classes=5)
    x = torch.randn(4, 1, 360)
    out = model(x)
    print(f"CNN1D output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")
