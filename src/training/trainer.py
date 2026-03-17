"""
Unified Training Engine
========================
Handles training loop, validation, early stopping, checkpointing,
and class-weighted loss for all model architectures.
Supports FP32 baseline and AMP (FP16/BF16) mixed precision.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def compute_class_weights(y, num_classes=5):
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = Counter(y.numpy() if isinstance(y, torch.Tensor) else y)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights.append(total / (num_classes * n))
    return torch.FloatTensor(weights)


class EarlyStopping:
    """Early stopping to terminate training when val loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    model_name="model",
    epochs=50,
    batch_size=128,
    lr=0.001,
    weight_decay=1e-4,
    patience=10,
    checkpoint_dir="results/checkpoints",
    precision="fp32",          # "fp32", "fp16", "bf16"
    device=None,
    num_classes=5,
):
    """
    Train a PyTorch model with optional mixed precision.

    Returns:
        dict with training history and final metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Class-weighted loss ───────────────────────────────────────────
    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── AMP setup ─────────────────────────────────────────────────────
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device.type == "cuda"))

    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")
    tag = f"{model_name}_{precision}"

    print(f"\n{'='*60}")
    print(f"  Training: {tag}")
    print(f"  Device: {device} | Precision: {precision}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)

                val_loss_sum += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct / val_total
        elapsed  = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        scheduler.step()

        # ── Checkpoint best ───────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, f"{tag}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)

        # ── Log ───────────────────────────────────────────────────────
        print(f"  Epoch {epoch:3d}/{epochs} │ "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
              f"{elapsed:.1f}s")

        # ── Early stopping ────────────────────────────────────────────
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # ── Save training history ─────────────────────────────────────────
    hist_path = os.path.join(checkpoint_dir, f"{tag}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Load best model ───────────────────────────────────────────────
    best_ckpt = torch.load(
        os.path.join(checkpoint_dir, f"{tag}_best.pt"),
        map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\n✅ Best val loss: {best_val_loss:.4f} at epoch {best_ckpt['epoch']}")
    print(f"   Checkpoint saved: {tag}_best.pt")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "best_val_acc": best_ckpt["val_acc"],
    }
