"""
1D Transformer for ECG Classification
=======================================
Temporal attention model with learned positional encoding.
Processes ECG beats as sequences of patches, then classifies
via a CLS token + FC head.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Transformer1D(nn.Module):
    """1D Transformer encoder for ECG beat classification."""

    def __init__(
        self,
        input_length=360,
        num_classes=5,
        in_channels=1,
        patch_size=10,       # split 360 samples into 36 patches
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size  # 36

        # Patch embedding: project each patch to d_model dimensions
        self.patch_embed = nn.Linear(in_channels * patch_size, d_model)

        # CLS token — learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 1,
                                          dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 1, 360)
            return_attention: if True, also return attention weights
        """
        B = x.size(0)

        # Reshape into patches: (B, num_patches, patch_size)
        x = x.view(B, -1, self.patch_size)         # (B, 36, 10)
        x = self.patch_embed(x)                      # (B, 36, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, 37, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        if return_attention:
            # Extract attention weights from each layer
            attentions = []
            for layer in self.encoder.layers:
                x_out, attn = layer.self_attn(x, x, x, need_weights=True)
                attentions.append(attn.detach())
                x = layer(x)
        else:
            x = self.encoder(x)

        # CLS token output → classifier
        cls_output = x[:, 0]  # (B, d_model)
        logits = self.classifier(cls_output)

        if return_attention:
            return logits, attentions
        return logits


if __name__ == "__main__":
    model = Transformer1D(input_length=360, num_classes=5)
    x = torch.randn(4, 1, 360)
    out = model(x)
    print(f"Transformer1D output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters          : {total:,}")

    # Test attention extraction
    logits, attn = model(x, return_attention=True)
    print(f"Attention layers: {len(attn)}, shape each: {attn[0].shape}")
