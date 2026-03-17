"""
Saliency Maps & Integrated Gradients
======================================
Gradient-based interpretability methods using Captum.
Implements: Vanilla Saliency, Integrated Gradients, SmoothGrad.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from captum.attr import (
        Saliency,
        IntegratedGradients,
        NoiseTunnel,
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠ captum not installed. Install with: pip install captum")


def compute_saliency(model, x, target_class=None):
    """Vanilla saliency map (input gradients)."""
    if not CAPTUM_AVAILABLE:
        return _manual_saliency(model, x, target_class)

    model.eval()
    saliency = Saliency(model)
    if target_class is None:
        target_class = model(x).argmax(1).item()
    attr = saliency.attribute(x, target=target_class, abs=True)
    return attr.squeeze().cpu().detach().numpy()


def compute_integrated_gradients(model, x, target_class=None, n_steps=50):
    """Integrated Gradients attribution."""
    if not CAPTUM_AVAILABLE:
        return _manual_saliency(model, x, target_class)

    model.eval()
    ig = IntegratedGradients(model)
    if target_class is None:
        target_class = model(x).argmax(1).item()
    attr = ig.attribute(x, target=target_class, n_steps=n_steps)
    return attr.squeeze().cpu().detach().numpy()


def compute_smoothgrad(model, x, target_class=None, nt_samples=20,
                       stdevs=0.1):
    """SmoothGrad: averaged gradients over noisy inputs."""
    if not CAPTUM_AVAILABLE:
        return _manual_saliency(model, x, target_class)

    model.eval()
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    if target_class is None:
        target_class = model(x).argmax(1).item()
    attr = nt.attribute(x, nt_type="smoothgrad", target=target_class,
                        nt_samples=nt_samples, stdevs=stdevs, abs=True)
    return attr.squeeze().cpu().detach().numpy()


def _manual_saliency(model, x, target_class=None):
    """Fallback vanilla saliency without captum."""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    if target_class is None:
        target_class = output.argmax(1).item()
    model.zero_grad()
    output[0, target_class].backward()
    return x.grad.abs().squeeze().cpu().detach().numpy()


def plot_saliency_comparison(signal, saliency_maps, method_names,
                              true_class, pred_class, class_names,
                              save_path=None, title=""):
    """
    Plot ECG signal with multiple saliency method overlays.
    """
    n = len(saliency_maps) + 1
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

    t = np.arange(len(signal))

    # Plot raw signal
    axes[0].plot(t, signal, color="black", lw=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"ECG Beat — True: {class_names[true_class]}, "
                      f"Pred: {class_names[pred_class]} {title}")

    for i, (sal, name) in enumerate(zip(saliency_maps, method_names)):
        ax = axes[i + 1]
        # Normalise saliency
        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        ax.plot(t, signal, color="gray", alpha=0.4, lw=0.5)
        im = ax.imshow(sal_norm[np.newaxis, :], aspect="auto", cmap="hot",
                       extent=[0, len(signal), signal.min(), signal.max()],
                       alpha=0.7)
        ax.set_ylabel(name)
        plt.colorbar(im, ax=ax, fraction=0.02)

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_saliency_analysis(model, X_test, y_test, class_names,
                           figures_dir, model_name="model",
                           precision_tag="fp32", n_samples=2):
    """Run all saliency methods on sample beats and save visualizations."""
    os.makedirs(figures_dir, exist_ok=True)
    device = next(model.parameters()).device

    for cls in range(len(class_names)):
        idx = np.where(y_test == cls)[0]
        if len(idx) == 0:
            continue

        for i, s_idx in enumerate(idx[:n_samples]):
            x = torch.FloatTensor(X_test[s_idx:s_idx+1]).to(device)
            pred = model(x).argmax(1).item()
            signal = X_test[s_idx, 0]

            # Compute attributions
            sal = compute_saliency(model, x, pred)
            ig  = compute_integrated_gradients(model, x, pred)
            sg  = compute_smoothgrad(model, x, pred)

            save_path = os.path.join(
                figures_dir,
                f"saliency_{model_name}_{precision_tag}_cls{cls}_s{i}.png"
            )
            plot_saliency_comparison(
                signal,
                [sal, ig, sg],
                ["Vanilla Saliency", "Integrated Gradients", "SmoothGrad"],
                cls, pred, class_names,
                save_path=save_path,
                title=f"[{model_name} {precision_tag}]"
            )

    print(f"  ✅ Saliency figures saved for {model_name}_{precision_tag}")
