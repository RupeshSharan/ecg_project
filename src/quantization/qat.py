"""
Quantization-Aware Training (QAT).

Fine-tunes a pretrained model with fake-quant nodes, converts to INT8,
and evaluates the quantized checkpoint.
"""

import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import evaluate_model, get_model_size_mb
from src.models.cnn1d import CNN1D
from src.quantization.ptq import QuantizableCNN1D, fuse_quantizable_cnn1d, load_pretrained
from src.training.trainer import compute_class_weights

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SPLITS = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
CKPT_DIR = os.path.join(PROJECT_ROOT, cfg["models"]["checkpoint_dir"])
FIG_DIR = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])
MET_DIR = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])


def qat_train(model_name="cnn1d", qat_epochs=10, lr=1e-4, batch_size=128):
    """
    Run QAT for CNN1D:
      1) load pretrained FP32 model
      2) fuse quantizable blocks
      3) prepare QAT + fine-tune
      4) convert to INT8
      5) evaluate on test split
    """
    device = torch.device("cpu")

    X_train = np.load(os.path.join(SPLITS, "X_train.npy"))
    y_train = np.load(os.path.join(SPLITS, "y_train.npy"))
    X_val = np.load(os.path.join(SPLITS, "X_val.npy"))
    y_val = np.load(os.path.join(SPLITS, "y_val.npy"))
    X_test = np.load(os.path.join(SPLITS, "X_test.npy"))
    y_test = np.load(os.path.join(SPLITS, "y_test.npy"))

    print(f"\n{'=' * 60}")
    print(f"  QAT: {model_name}")
    print(f"{'=' * 60}")

    base_model = load_pretrained(CNN1D, model_name, device, input_length=360, num_classes=5)
    if base_model is None:
        return None

    qat_model = QuantizableCNN1D(copy.deepcopy(base_model))
    qat_model = fuse_quantizable_cnn1d(qat_model)
    qat_model.train()

    qat_model.qconfig = quant.get_default_qat_qconfig("x86")
    quant.prepare_qat(qat_model, inplace=True)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    class_weights = compute_class_weights(y_train, 5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=qat_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = copy.deepcopy(qat_model.state_dict())

    for epoch in range(1, qat_epochs + 1):
        qat_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = qat_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        qat_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = qat_model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"  Epoch {epoch:2d}/{qat_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(qat_model.state_dict())

    qat_model.load_state_dict(best_state)
    qat_model.eval()
    quant.convert(qat_model, inplace=True)

    q_size = get_model_size_mb(qat_model)
    print(f"\n  QAT complete - quantized model size: {q_size:.2f} MB")

    torch.save(qat_model.state_dict(), os.path.join(CKPT_DIR, f"{model_name}_qat_int8_best.pt"))

    metrics = evaluate_model(
        model=qat_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        precision_tag="qat_int8",
        class_names=cfg["data"]["class_names"],
        figures_dir=FIG_DIR,
        metrics_dir=MET_DIR,
        device=device,
    )
    return metrics


def main():
    results = {}
    metrics = qat_train(model_name="cnn1d", qat_epochs=10, lr=1e-4)
    if metrics:
        results["cnn1d_qat_int8"] = metrics

    with open(os.path.join(MET_DIR, "qat_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nQAT experiments complete!")


if __name__ == "__main__":
    main()
