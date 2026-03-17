"""
Ablation studies:
1) Quantization granularity (per-tensor vs per-channel)
2) Calibration dataset size sweep for PTQ
3) Faithfulness tests (deletion/insertion)
"""

import copy
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.quantization as quant
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import evaluate_model
from src.interpretability.saliency import compute_saliency
from src.quantization.ptq import QuantizableCNN1D, fuse_quantizable_cnn1d


def _quantize_static(model, X_cal, qconfig=None, batch_size=64):
    """
    Build a quantized static INT8 model from FP32 base model and calibration data.
    """
    q_model = QuantizableCNN1D(copy.deepcopy(model))
    q_model = fuse_quantizable_cnn1d(q_model)
    q_model.eval()
    q_model.qconfig = qconfig if qconfig is not None else quant.get_default_qconfig("x86")
    quant.prepare(q_model, inplace=True)

    cal_loader = DataLoader(TensorDataset(torch.FloatTensor(X_cal)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in cal_loader:
            _ = q_model(xb)

    quant.convert(q_model, inplace=True)
    return q_model


def compare_quantization_granularity(
    model,
    X_cal,
    X_test,
    y_test,
    class_names,
    figures_dir,
    metrics_dir,
):
    """Compare per-tensor vs per-channel weight quantization."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ONEDNN requires symmetric weight quantization for conv prepack.
    configs = {
        "per_tensor": quant.QConfig(
            activation=quant.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            ),
            weight=quant.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
            ),
        ),
        "per_channel": quant.QConfig(
            activation=quant.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            ),
            weight=quant.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            ),
        ),
    }

    results = {}
    for name, qconfig in configs.items():
        print(f"\n  Quantization: {name}")
        q_model = _quantize_static(model, X_cal, qconfig=qconfig)

        metrics = evaluate_model(
            model=q_model,
            X_test=X_test,
            y_test=y_test,
            model_name="cnn1d",
            precision_tag=f"int8_{name}",
            class_names=class_names,
            figures_dir=figures_dir,
            metrics_dir=metrics_dir,
            device=torch.device("cpu"),
        )
        results[name] = metrics

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.15, accs, 0.3, label="Accuracy", color="steelblue")
    ax.bar(x + 0.15, f1s, 0.3, label="F1 Macro", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in names])
    ax.set_ylabel("Score")
    ax.set_title("Quantization Granularity: Per-Tensor vs Per-Channel")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ablation_granularity.png"), dpi=150)
    plt.close()

    return results


def calibration_size_sweep(
    model,
    X_train,
    X_test,
    y_test,
    class_names,
    figures_dir,
    metrics_dir,
    sizes=(10, 50, 100, 250, 500, 1000, 2000),
):
    """Vary calibration dataset size for PTQ and measure accuracy."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    results = {}
    for cal_size in sizes:
        n = min(cal_size, len(X_train))
        idx = np.random.choice(len(X_train), n, replace=False)
        X_cal = X_train[idx]

        q_model = _quantize_static(model, X_cal)
        metrics = evaluate_model(
            model=q_model,
            X_test=X_test,
            y_test=y_test,
            model_name="cnn1d",
            precision_tag=f"int8_cal{n}",
            class_names=class_names,
            figures_dir=figures_dir,
            metrics_dir=metrics_dir,
            device=torch.device("cpu"),
        )
        results[n] = metrics["accuracy"]
        print(f"    Cal size {n:5d} -> Accuracy: {metrics['accuracy']:.4f}")

    sizes_list = sorted(results.keys())
    accs_list = [results[s] for s in sizes_list]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes_list, accs_list, "o-", color="steelblue", lw=2, markersize=8)
    ax.set_xlabel("Calibration Dataset Size")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Effect of Calibration Data Size on PTQ Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ablation_calibration_size.png"), dpi=150)
    plt.close()

    return results


def faithfulness_deletion_test(model, X_test, y_test, n_samples=50, n_steps=10, device="cpu"):
    """
    Deletion test: progressively mask most important features and track confidence drop.
    Lower AUC is better.
    """
    model = model.to(device)
    model.eval()

    sample_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    step_fracs = np.linspace(0, 1, n_steps + 1)
    all_curves = []

    for idx in sample_idx:
        x = torch.FloatTensor(X_test[idx : idx + 1]).to(device)
        pred = model(x).argmax(1).item()
        sal = compute_saliency(model, x, pred).flatten()
        order = np.argsort(-sal)
        signal_len = len(sal)

        scores = []
        for frac in step_fracs:
            n_mask = int(frac * signal_len)
            x_masked = x.clone()
            if n_mask > 0:
                mask_idx = order[:n_mask]
                x_masked[0, 0, mask_idx] = 0.0
            with torch.no_grad():
                prob = torch.softmax(model(x_masked), dim=1)[0, pred].item()
            scores.append(prob)

        all_curves.append(scores)

    return step_fracs, np.array(all_curves)


def faithfulness_insertion_test(model, X_test, y_test, n_samples=50, n_steps=10, device="cpu"):
    """
    Insertion test: progressively reveal most important features and track confidence rise.
    Higher AUC is better.
    """
    model = model.to(device)
    model.eval()

    sample_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    step_fracs = np.linspace(0, 1, n_steps + 1)
    all_curves = []

    for idx in sample_idx:
        x = torch.FloatTensor(X_test[idx : idx + 1]).to(device)
        pred = model(x).argmax(1).item()
        sal = compute_saliency(model, x, pred).flatten()
        order = np.argsort(-sal)
        signal_len = len(sal)

        scores = []
        for frac in step_fracs:
            n_reveal = int(frac * signal_len)
            x_partial = torch.zeros_like(x)
            if n_reveal > 0:
                reveal_idx = order[:n_reveal]
                x_partial[0, 0, reveal_idx] = x[0, 0, reveal_idx]
            with torch.no_grad():
                prob = torch.softmax(model(x_partial), dim=1)[0, pred].item()
            scores.append(prob)

        all_curves.append(scores)

    return step_fracs, np.array(all_curves)


def run_faithfulness_analysis(
    model,
    X_test,
    y_test,
    class_names,
    figures_dir,
    metrics_dir,
    model_name="model",
    precision_tag="fp32",
):
    """Run deletion and insertion faithfulness tests and save plots/metrics."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    device = next(model.parameters()).device

    del_fracs, del_curves = faithfulness_deletion_test(model, X_test, y_test, device=str(device))
    del_mean = del_curves.mean(axis=0)
    del_auc = np.trapz(del_mean, del_fracs)

    ins_fracs, ins_curves = faithfulness_insertion_test(model, X_test, y_test, device=str(device))
    ins_mean = ins_curves.mean(axis=0)
    ins_auc = np.trapz(ins_mean, ins_fracs)

    print(f"  Deletion AUC  : {del_auc:.4f} (lower = more faithful)")
    print(f"  Insertion AUC : {ins_auc:.4f} (higher = more faithful)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(del_fracs, del_mean, "r-o", lw=2, label=f"AUC={del_auc:.3f}")
    ax1.fill_between(
        del_fracs,
        del_curves.mean(0) - del_curves.std(0),
        del_curves.mean(0) + del_curves.std(0),
        alpha=0.2,
        color="red",
    )
    ax1.set_xlabel("Fraction of Features Deleted")
    ax1.set_ylabel("Prediction Confidence")
    ax1.set_title(f"Deletion Test [{model_name} {precision_tag}]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ins_fracs, ins_mean, "b-o", lw=2, label=f"AUC={ins_auc:.3f}")
    ax2.fill_between(
        ins_fracs,
        ins_curves.mean(0) - ins_curves.std(0),
        ins_curves.mean(0) + ins_curves.std(0),
        alpha=0.2,
        color="blue",
    )
    ax2.set_xlabel("Fraction of Features Inserted")
    ax2.set_ylabel("Prediction Confidence")
    ax2.set_title(f"Insertion Test [{model_name} {precision_tag}]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, f"faithfulness_{model_name}_{precision_tag}.png"),
        dpi=150,
    )
    plt.close()

    result = {"deletion_auc": float(del_auc), "insertion_auc": float(ins_auc)}
    with open(
        os.path.join(metrics_dir, f"faithfulness_{model_name}_{precision_tag}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result
