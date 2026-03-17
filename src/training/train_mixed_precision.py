"""
Mixed Precision Training Script
=================================
Trains all models with FP16 and BF16 precision using PyTorch AMP.
Compares convergence speed, accuracy, memory usage against FP32.
"""

import os
import sys
import yaml
import json
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

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")) as f:
    cfg = yaml.safe_load(f)

SPLITS   = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
CKPT_DIR = os.path.join(PROJECT_ROOT, cfg["models"]["checkpoint_dir"])
FIG_DIR  = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])
MET_DIR  = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])
SEED     = cfg["training"]["seed"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(name):
    """Factory to create fresh model instance."""
    factory = {
        "cnn1d":       lambda: CNN1D(input_length=360, num_classes=5),
        "resnet1d":    lambda: ResNet1D(input_length=360, num_classes=5),
        "bilstm":      lambda: BiLSTM(input_length=360, num_classes=5),
        "transformer": lambda: Transformer1D(input_length=360, num_classes=5),
    }
    return factory[name]()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    if device.type != "cuda":
        print("⚠ Mixed precision training benefits most from GPU.")
        print("  Running on CPU — FP16 will use emulation (slower).")

    # ── Load data ─────────────────────────────────────────────────────
    X_train = np.load(os.path.join(SPLITS, "X_train.npy"))
    y_train = np.load(os.path.join(SPLITS, "y_train.npy"))
    X_val   = np.load(os.path.join(SPLITS, "X_val.npy"))
    y_val   = np.load(os.path.join(SPLITS, "y_val.npy"))
    X_test  = np.load(os.path.join(SPLITS, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS, "y_test.npy"))

    model_names = ["cnn1d", "resnet1d", "bilstm", "transformer"]

    # Determine available precision formats
    precisions = ["fp16"]
    if device.type == "cuda":
        # BF16 requires Ampere or newer GPU
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16")
            print("✅ BF16 supported on this GPU")
        else:
            print("ℹ  BF16 not supported — skipping BF16 experiments")
    else:
        print("ℹ  Running on CPU — FP16 only (emulated)")

    all_metrics = {}

    for prec in precisions:
        for name in model_names:
            set_seed(SEED)
            model = get_model(name)
            tag = f"{name}_{prec}"

            print(f"\n{'#'*60}")
            print(f"  Training: {tag}")
            print(f"{'#'*60}")

            # Track GPU memory if available
            mem_before = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1e6

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
                precision=prec,
                device=device,
            )

            # Record peak memory
            peak_mem = 0
            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / 1e6

            plot_training_history(result["history"], name, prec, FIG_DIR)

            metrics = evaluate_model(
                model=result["model"],
                X_test=X_test, y_test=y_test,
                model_name=name,
                precision_tag=prec,
                class_names=cfg["data"]["class_names"],
                figures_dir=FIG_DIR,
                metrics_dir=MET_DIR,
                device=device,
            )

            # Add memory and timing info
            metrics["peak_gpu_memory_mb"] = peak_mem
            metrics["avg_epoch_time_s"] = np.mean(result["history"]["epoch_time"])
            metrics["total_training_time_s"] = sum(result["history"]["epoch_time"])

            all_metrics[tag] = metrics

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  MIXED PRECISION RESULTS")
    print(f"{'='*90}")
    print(f"{'Tag':>20s} | {'Acc':>7s} | {'F1':>7s} | {'AUC':>7s} | "
          f"{'Epoch(s)':>8s} | {'VRAM(MB)':>8s} | {'Lat(ms)':>8s}")
    print("-" * 90)
    for tag, m in all_metrics.items():
        print(f"{tag:>20s} | {m['accuracy']:7.4f} | {m['f1_macro']:7.4f} | "
              f"{m['auc_roc']:7.4f} | {m['avg_epoch_time_s']:8.2f} | "
              f"{m.get('peak_gpu_memory_mb', 0):8.1f} | "
              f"{m['latency_ms']['mean_ms']:8.2f}")

    # Save combined results
    with open(os.path.join(MET_DIR, "mixed_precision_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n✅ Mixed precision experiments complete!")


if __name__ == "__main__":
    main()
