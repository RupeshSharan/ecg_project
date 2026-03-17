"""
Model Evaluator
===============
Comprehensive evaluation: accuracy, class-wise metrics, AUC-ROC,
confusion matrix, inference latency, and model size.
"""

import io
import json
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def get_model_size_mb(model):
    """Estimate serialized model size in MB, robust for quantized models."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    tensor_bytes = param_size + buffer_size

    # Quantized models keep packed params in state_dict entries.
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    state_dict_bytes = buf.getbuffer().nbytes

    return max(tensor_bytes, state_dict_bytes) / (1024**2)


def measure_latency(model, input_shape=(1, 1, 360), device=None, n_runs=100):
    """Measure average inference latency (ms)."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name="model",
    precision_tag="fp32",
    class_names=None,
    figures_dir="results/figures",
    metrics_dir="results/metrics",
    device=None,
    batch_size=256,
):
    """
    Full evaluation and artifact export.
    Returns a dict of metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if class_names is None:
        class_names = ["Normal", "SVEB", "VEB", "Fusion", "Paced"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    latency = measure_latency(model, device=device)
    size_mb = get_model_size_mb(model)
    total_params = sum(p.numel() for p in model.parameters())

    tag = f"{model_name}_{precision_tag}"
    metrics = {
        "model": model_name,
        "precision": precision_tag,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "auc_roc": float(auc),
        "f1_per_class": {class_names[i]: float(f1_per[i]) for i in range(num_classes)},
        "latency_ms": latency,
        "model_size_mb": float(size_mb),
        "total_params": int(total_params),
    }

    with open(os.path.join(metrics_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {tag}", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{tag}_confusion.png"), dpi=150)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(metrics_dir, f"{tag}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'=' * 50}")
    print(f"  Evaluation: {tag}")
    print(f"{'=' * 50}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 (macro)   : {f1_macro:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Latency      : {latency['mean_ms']:.2f} +/- {latency['std_ms']:.2f} ms")
    print(f"  Model size   : {size_mb:.2f} MB")
    print(f"\n{report}")

    return metrics


def plot_training_history(history, model_name, precision_tag, figures_dir="results/figures"):
    """Plot training/validation loss and accuracy curves."""
    os.makedirs(figures_dir, exist_ok=True)
    tag = f"{model_name}_{precision_tag}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title(f"{tag} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_title(f"{tag} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{tag}_curves.png"), dpi=150)
    plt.close()
    print(f"  Training curves saved: {tag}_curves.png")
