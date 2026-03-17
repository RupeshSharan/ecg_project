"""
Train All Baseline Models (FP32)
=================================
Trains 1D-CNN, ResNet1D, BiLSTM, and Transformer1D on FP32,
evaluates each on the test set, and saves checkpoints + metrics.
"""

import os
import sys
import yaml
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn1d import CNN1D
from src.models.resnet1d import ResNet1D
from src.models.bilstm import BiLSTM
from src.models.transformer1d import Transformer1D
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model, plot_training_history

# ── Load config ───────────────────────────────────────────────────────
with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")) as f:
    cfg = yaml.safe_load(f)

SPLITS    = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
CKPT_DIR  = os.path.join(PROJECT_ROOT, cfg["models"]["checkpoint_dir"])
FIG_DIR   = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])
MET_DIR   = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])
SEED      = cfg["training"]["seed"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    X_train = np.load(os.path.join(SPLITS, "X_train.npy"))
    y_train = np.load(os.path.join(SPLITS, "y_train.npy"))
    X_val   = np.load(os.path.join(SPLITS, "X_val.npy"))
    y_val   = np.load(os.path.join(SPLITS, "y_val.npy"))
    X_test  = np.load(os.path.join(SPLITS, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS, "y_test.npy"))

    print(f"📊 Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Model registry ────────────────────────────────────────────────
    models = {
        "cnn1d":       CNN1D(input_length=360, num_classes=5),
        "resnet1d":    ResNet1D(input_length=360, num_classes=5),
        "bilstm":      BiLSTM(input_length=360, num_classes=5),
        "transformer": Transformer1D(input_length=360, num_classes=5),
    }

    all_metrics = {}

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'#'*60}")
        print(f"  Model: {name} | Params: {total_params:,}")
        print(f"{'#'*60}")

        # Train
        result = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            model_name=name,
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            patience=cfg["training"]["patience"],
            checkpoint_dir=CKPT_DIR,
            precision="fp32",
            device=device,
        )

        # Plot training curves
        plot_training_history(result["history"], name, "fp32", FIG_DIR)

        # Evaluate on test set
        metrics = evaluate_model(
            model=result["model"],
            X_test=X_test, y_test=y_test,
            model_name=name,
            precision_tag="fp32",
            class_names=cfg["data"]["class_names"],
            figures_dir=FIG_DIR,
            metrics_dir=MET_DIR,
            device=device,
        )

        all_metrics[name] = metrics

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  BASELINE RESULTS SUMMARY (FP32)")
    print(f"{'='*80}")
    print(f"{'Model':>15s} | {'Acc':>7s} | {'F1':>7s} | {'AUC':>7s} | "
          f"{'Latency':>10s} | {'Size (MB)':>9s} | {'Params':>10s}")
    print("-" * 80)
    for name, m in all_metrics.items():
        print(f"{name:>15s} | {m['accuracy']:7.4f} | {m['f1_macro']:7.4f} | "
              f"{m['auc_roc']:7.4f} | {m['latency_ms']['mean_ms']:8.2f}ms | "
              f"{m['model_size_mb']:9.2f} | {m['total_params']:>10,}")

    print(f"\n✅ All baselines trained and evaluated!")
    print(f"   Checkpoints → {CKPT_DIR}/")
    print(f"   Metrics     → {MET_DIR}/")
    print(f"   Figures     → {FIG_DIR}/")


if __name__ == "__main__":
    main()
