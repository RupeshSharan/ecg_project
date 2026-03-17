"""
Attention visualization for Transformer models.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def extract_attention_weights(model, x):
    """
    Extract attention weights from Transformer1D.
    Returns:
      logits
      attentions: list of attention tensors (shape may vary by PyTorch version)
    """
    model.eval()
    with torch.no_grad():
        logits, attentions = model(x, return_attention=True)
    return logits, attentions


def _to_2d_attention(attn_tensor, head_idx=None):
    """
    Normalize attention tensor to a 2D map [query, key].
    Supports:
      - [heads, q, k]
      - [q, k]
      - [q] (degraded fallback)
    """
    a = np.asarray(attn_tensor.detach().cpu())

    # Remove single batch dimensions if present.
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]

    if a.ndim == 3:
        if head_idx is not None and 0 <= head_idx < a.shape[0]:
            return a[head_idx]
        return a.mean(axis=0)

    if a.ndim == 2:
        return a

    if a.ndim == 1:
        return np.expand_dims(a, axis=0)

    raise ValueError(f"Unsupported attention shape: {a.shape}")


def plot_attention_heatmap(
    attentions,
    signal,
    true_class,
    pred_class,
    class_names,
    layer_idx=-1,
    head_idx=None,
    save_path=None,
    model_name="transformer",
):
    """Plot attention heatmap and ECG signal."""
    attn_map = _to_2d_attention(attentions[layer_idx], head_idx=head_idx)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 0.15], height_ratios=[1, 3])

    ax_sig = fig.add_subplot(gs[0, 0])
    ax_sig.plot(signal, color="black", lw=0.8)
    ax_sig.set_title(
        f"Attention - Layer {layer_idx} | True: {class_names[true_class]}, "
        f"Pred: {class_names[pred_class]} [{model_name}]"
    )
    ax_sig.set_ylabel("Amplitude")
    ax_sig.set_xlim(0, len(signal))

    ax_hm = fig.add_subplot(gs[1, 0])
    sns.heatmap(attn_map, ax=ax_hm, cmap="viridis", cbar=False)
    ax_hm.set_xlabel("Key Position (patch)")
    ax_hm.set_ylabel("Query Position (patch)")

    ax_cb = fig.add_subplot(gs[1, 1])
    plt.colorbar(ax_hm.collections[0], cax=ax_cb)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cls_attention(
    attentions,
    signal,
    true_class,
    pred_class,
    class_names,
    patch_size=10,
    save_path=None,
    model_name="transformer",
):
    """
    Plot CLS token attention over ECG patches, upsampled to sample resolution.
    """
    cls_rows = []
    for attn in attentions:
        a2d = _to_2d_attention(attn, head_idx=None)
        if a2d.shape[-1] <= 1:
            continue
        # CLS token row (index 0), excluding self-column.
        cls_rows.append(a2d[0, 1:])

    if not cls_rows:
        return

    avg_cls_attn = np.mean(cls_rows, axis=0)
    attn_signal = np.repeat(avg_cls_attn, patch_size)[: len(signal)]
    attn_signal = (attn_signal - attn_signal.min()) / (
        attn_signal.max() - attn_signal.min() + 1e-8
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    t = np.arange(len(signal))

    ax1.plot(t, signal, color="black", lw=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"CLS Token Attention - True: {class_names[true_class]}, "
        f"Pred: {class_names[pred_class]} [{model_name}]"
    )

    ax2.fill_between(t, 0, attn_signal, alpha=0.5, color="purple")
    ax2.plot(t, attn_signal, color="purple", lw=1.0)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("CLS Attention")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_attention_analysis(
    model,
    X_test,
    y_test,
    class_names,
    figures_dir,
    model_name="transformer",
    precision_tag="fp32",
    n_samples=2,
    patch_size=10,
):
    """Run attention visualization on sample beats."""
    os.makedirs(figures_dir, exist_ok=True)
    device = next(model.parameters()).device

    for cls in range(len(class_names)):
        idx = np.where(y_test == cls)[0]
        if len(idx) == 0:
            continue

        for i, s_idx in enumerate(idx[:n_samples]):
            x = torch.FloatTensor(X_test[s_idx : s_idx + 1]).to(device)
            logits, attentions = extract_attention_weights(model, x)
            pred = logits.argmax(1).item()
            signal = X_test[s_idx, 0]

            save_hm = os.path.join(
                figures_dir,
                f"attn_heatmap_{model_name}_{precision_tag}_cls{cls}_s{i}.png",
            )
            plot_attention_heatmap(
                attentions,
                signal,
                cls,
                pred,
                class_names,
                save_path=save_hm,
                model_name=f"{model_name} {precision_tag}",
            )

            save_cls = os.path.join(
                figures_dir,
                f"attn_cls_{model_name}_{precision_tag}_cls{cls}_s{i}.png",
            )
            plot_cls_attention(
                attentions,
                signal,
                cls,
                pred,
                class_names,
                patch_size=patch_size,
                save_path=save_cls,
                model_name=f"{model_name} {precision_tag}",
            )

    print(f"  Attention figures saved for {model_name}_{precision_tag}")
