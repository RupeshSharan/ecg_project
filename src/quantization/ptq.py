"""
Post-Training Quantization (PTQ).

Applies dynamic and static INT8 quantization to trained FP32 models.
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import evaluate_model, get_model_size_mb
from src.models.cnn1d import CNN1D
from src.models.resnet1d import ResNet1D

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SPLITS = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
CKPT_DIR = os.path.join(PROJECT_ROOT, cfg["models"]["checkpoint_dir"])
FIG_DIR = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])
MET_DIR = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])
CAL_SIZE = cfg["quantization"]["calibration_samples"]


def load_pretrained(model_cls, model_name, device="cpu", **kwargs):
    """Load a pretrained FP32 model checkpoint."""
    model = model_cls(**kwargs)
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fp32_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def dynamic_quantize(model):
    """
    Dynamic INT8 quantization (weights quantized, activations at runtime).
    Best for Linear/LSTM-heavy subgraphs.
    """
    return torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)


class QuantizableCNN1D(nn.Module):
    """CNN1D wrapper with quant/dequant stubs for eager-mode quantization."""

    def __init__(self, base_model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = base_model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.model.classifier(x)
        x = self.dequant(x)
        return x


def fuse_quantizable_cnn1d(model):
    """
    Fuse Conv-BN-ReLU and Linear-ReLU blocks.
    Required so quantized inference does not invoke BatchNorm kernels.
    """
    model.eval()
    quant.fuse_modules(
        model,
        [
            ["model.features.0", "model.features.1", "model.features.2"],
            ["model.features.5", "model.features.6", "model.features.7"],
            ["model.features.10", "model.features.11", "model.features.12"],
            ["model.classifier.0", "model.classifier.1"],
            ["model.classifier.3", "model.classifier.4"],
        ],
        inplace=True,
    )
    return model


def static_quantize(model, X_cal):
    """
    Static INT8 quantization with calibration.
    Args:
      model: FP32 CNN1D model
      X_cal: calibration samples, shape (N, 1, 360)
    """
    q_model = QuantizableCNN1D(copy.deepcopy(model))
    q_model = fuse_quantizable_cnn1d(q_model)
    q_model.qconfig = quant.get_default_qconfig("x86")
    quant.prepare(q_model, inplace=True)

    cal_loader = DataLoader(TensorDataset(torch.FloatTensor(X_cal)), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (xb,) in cal_loader:
            _ = q_model(xb)

    quant.convert(q_model, inplace=True)
    return q_model


def main():
    device = torch.device("cpu")
    print("Post-Training Quantization (PTQ)\n")

    X_test = np.load(os.path.join(SPLITS, "X_test.npy"))
    y_test = np.load(os.path.join(SPLITS, "y_test.npy"))
    X_train = np.load(os.path.join(SPLITS, "X_train.npy"))

    np.random.seed(42)
    cal_idx = np.random.choice(len(X_train), min(CAL_SIZE, len(X_train)), replace=False)
    X_cal = X_train[cal_idx]

    results = {}
    model_configs = [("cnn1d", CNN1D, {"input_length": 360, "num_classes": 5})]

    for model_name, model_cls, kwargs in model_configs:
        print(f"\n{'=' * 50}")
        print(f"  Quantizing: {model_name}")
        print(f"{'=' * 50}")

        model = load_pretrained(model_cls, model_name, device, **kwargs)
        if model is None:
            continue

        fp32_size = get_model_size_mb(model)

        print("\n  -> Dynamic INT8 quantization...")
        dyn_model = dynamic_quantize(copy.deepcopy(model))
        dyn_size = get_model_size_mb(dyn_model)
        print(
            f"     FP32 size: {fp32_size:.2f} MB -> "
            f"Dynamic INT8: {dyn_size:.2f} MB ({dyn_size / fp32_size * 100:.1f}%)"
        )

        metrics_dyn = evaluate_model(
            model=dyn_model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            precision_tag="int8_dynamic",
            class_names=cfg["data"]["class_names"],
            figures_dir=FIG_DIR,
            metrics_dir=MET_DIR,
            device=device,
        )
        results[f"{model_name}_int8_dynamic"] = metrics_dyn

        print("\n  -> Static INT8 quantization (with calibration)...")
        try:
            stat_model = static_quantize(model, X_cal)
            stat_size = get_model_size_mb(stat_model)
            print(
                f"     FP32 size: {fp32_size:.2f} MB -> "
                f"Static INT8: {stat_size:.2f} MB ({stat_size / fp32_size * 100:.1f}%)"
            )

            metrics_stat = evaluate_model(
                model=stat_model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                precision_tag="int8_static",
                class_names=cfg["data"]["class_names"],
                figures_dir=FIG_DIR,
                metrics_dir=MET_DIR,
                device=device,
            )
            results[f"{model_name}_int8_static"] = metrics_stat
        except Exception as exc:
            print(f"     Static quantization failed: {exc}")
            print("     Continuing with dynamic quantization only.")

    print(f"\n{'=' * 50}")
    print("  Quantizing: resnet1d (Dynamic INT8)")
    print(f"{'=' * 50}")
    resnet = load_pretrained(ResNet1D, "resnet1d", device, input_length=360, num_classes=5)
    if resnet is not None:
        dyn_resnet = dynamic_quantize(copy.deepcopy(resnet))
        metrics = evaluate_model(
            model=dyn_resnet,
            X_test=X_test,
            y_test=y_test,
            model_name="resnet1d",
            precision_tag="int8_dynamic",
            class_names=cfg["data"]["class_names"],
            figures_dir=FIG_DIR,
            metrics_dir=MET_DIR,
            device=device,
        )
        results["resnet1d_int8_dynamic"] = metrics

    with open(os.path.join(MET_DIR, "ptq_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nPTQ experiments complete. Results in {MET_DIR}{os.sep}")


if __name__ == "__main__":
    main()
