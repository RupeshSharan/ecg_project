"""
Bidirectional LSTM for ECG Classification
==========================================
Sequence model that processes the ECG beat as a time series.
Uses bidirectional LSTM layers followed by an FC classifier.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM for single-lead ECG beat classification."""

    def __init__(self, input_length=360, num_classes=5, in_channels=1,
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_length = input_length
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers

        # Optional 1D conv front-end to reduce sequence length
        self.conv_front = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Compute sequence length after conv (input: 360 → 90)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            conv_out = self.conv_front(dummy)  # (1, 64, L')
            self.seq_len  = conv_out.shape[2]
            self.feat_dim = conv_out.shape[1]

        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 360)
        x = self.conv_front(x)          # (B, 64, L')
        x = x.permute(0, 2, 1)          # (B, L', 64) — seq format
        lstm_out, _ = self.lstm(x)       # (B, L', hidden*2)
        # Use last time-step output
        x = lstm_out[:, -1, :]           # (B, hidden*2)
        return self.classifier(x)


if __name__ == "__main__":
    model = BiLSTM(input_length=360, num_classes=5)
    x = torch.randn(4, 1, 360)
    out = model(x)
    print(f"BiLSTM output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters   : {total:,}")
