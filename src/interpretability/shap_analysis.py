"""
SHAP analysis for ECG models.

Supports DeepSHAP with a robust fallback to KernelSHAP.
"""

import copy
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Install with: pip install shap")


def _make_shap_safe_model(model, device="cpu"):
    """Clone model and disable inplace activations for safer gradient hooks."""
    safe_model = copy.deepcopy(model).to(device)
    safe_model.eval()
    for module in safe_model.modules():
        if isinstance(module, nn.ReLU) and getattr(module, "inplace", False):
            module.inplace = False
    return safe_model


class PyTorchWrapper:
    """Wrap a PyTorch classifier for SHAP KernelExplainer."""

    def __init__(self, model, device="cpu", signal_len=360):
        self.model = model
        self.device = device
        self.signal_len = signal_len
        self.model.eval()

    def _to_model_input(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, self.signal_len)
        if x.ndim == 2:
            if x.shape[1] != self.signal_len:
                raise ValueError(f"Expected flattened signal length {self.signal_len}, got {x.shape}")
            x = x[:, np.newaxis, :]
        if x.ndim != 3:
            raise ValueError(f"Expected 1D/2D/3D input, got shape {x.shape}")
        return torch.from_numpy(x).to(self.device)

    def __call__(self, x):
        x_tensor = self._to_model_input(x)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def _normalize_shap_output(shap_values, n_classes):
    """
    Normalize SHAP output to a list[class] -> array[n_samples, signal_len].
    Handles both list and ndarray return types.
    """
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        # (n_samples, signal_len)
        return [sv for _ in range(n_classes)]
    if sv.ndim == 3:
        # Could be (n_samples, signal_len, n_classes) or (n_classes, n_samples, signal_len)
        if sv.shape[-1] == n_classes:
            return [sv[:, :, c] for c in range(n_classes)]
        if sv.shape[0] == n_classes:
            return [sv[c] for c in range(n_classes)]
    # Last resort: return one map duplicated.
    flat = sv.reshape(sv.shape[0], -1)
    return [flat for _ in range(n_classes)]


def compute_deep_shap(model, X_test, X_background, class_names, n_samples=5, device="cpu"):
    """
    Compute SHAP values for test samples with DeepSHAP and KernelSHAP fallback.
    Returns:
      shap_values_by_class: list of arrays [n_samples, signal_len]
      test_samples: array [n_samples, 1, signal_len]
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available")
        return None, None

    n_classes = len(class_names)
    signal_len = X_test.shape[-1]
    n_samples = min(n_samples, len(X_test))

    model_safe = _make_shap_safe_model(model, device=device)
    bg = torch.FloatTensor(X_background).to(device)
    test = torch.FloatTensor(X_test[:n_samples]).to(device)

    try:
        explainer = shap.DeepExplainer(model_safe, bg)
        shap_values = explainer.shap_values(test)
        return _normalize_shap_output(shap_values, n_classes), X_test[:n_samples]
    except Exception as exc:
        print(f"  Warning: DeepSHAP failed ({exc}), falling back to KernelSHAP...")

    try:
        wrapper = PyTorchWrapper(model_safe, device=device, signal_len=signal_len)
        bg_np = X_background[:50].reshape(min(50, len(X_background)), signal_len)
        test_flat = X_test[:n_samples].reshape(n_samples, signal_len)

        explainer = shap.KernelExplainer(wrapper, bg_np)
        shap_values = explainer.shap_values(test_flat, nsamples=100)
        return _normalize_shap_output(shap_values, n_classes), X_test[:n_samples]
    except Exception as exc:
        print(f"  Warning: KernelSHAP failed ({exc}). Skipping SHAP analysis.")
        return None, None


def plot_shap_waterfall(shap_values, signal, class_idx, class_names, save_path=None):
    """Plot SHAP values overlaid on ECG signal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    t = np.arange(len(signal))
    ax1.plot(t, signal, color="black", lw=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"ECG Beat - SHAP for class: {class_names[class_idx]}")

    sv = np.asarray(shap_values).reshape(-1)[: len(signal)]
    colors = ["red" if v > 0 else "blue" for v in sv]
    ax2.bar(t, sv, color=colors, alpha=0.7, width=1.0)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("SHAP Value")
    ax2.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_shap_analysis(
    model,
    X_test,
    y_test,
    X_train,
    class_names,
    figures_dir,
    model_name="model",
    precision_tag="fp32",
    n_samples=3,
    n_bg=100,
):
    """Run SHAP analysis and save visualizations."""
    if not SHAP_AVAILABLE:
        print("  Skipping SHAP analysis - shap not installed")
        return

    os.makedirs(figures_dir, exist_ok=True)
    device = next(model.parameters()).device

    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), min(n_bg, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    shap_values_by_class, test_samples = compute_deep_shap(
        model,
        X_test,
        X_bg,
        class_names,
        n_samples=n_samples,
        device=str(device),
    )

    if shap_values_by_class is None or test_samples is None:
        return

    n_samples_eff = min(n_samples, len(test_samples))
    for i in range(n_samples_eff):
        signal = test_samples[i, 0] if test_samples[i].ndim > 1 else test_samples[i]
        for cls in range(len(class_names)):
            if cls >= len(shap_values_by_class):
                continue
            cls_values = np.asarray(shap_values_by_class[cls])
            if i >= len(cls_values):
                continue

            save_path = os.path.join(
                figures_dir,
                f"shap_{model_name}_{precision_tag}_s{i}_cls{cls}.png",
            )
            plot_shap_waterfall(cls_values[i], signal, cls, class_names, save_path)

    print(f"  SHAP figures saved for {model_name}_{precision_tag}")
