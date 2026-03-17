"""
Cross-Precision Explanation Comparison
========================================
Compare interpretability explanations across FP32, FP16, INT8, INT4
versions of the same model to quantify "explanation drift" caused by
quantization and mixed precision.
"""

import os
import sys
import json
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.interpretability.saliency import (
    compute_saliency,
    compute_integrated_gradients,
)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = a.flatten(), b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return 1.0 - cosine_dist(a, b)


def spearman_correlation(a, b):
    """Compute Spearman rank correlation."""
    a, b = a.flatten(), b.flatten()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr, _ = spearmanr(a, b)
    return float(corr)


def compare_explanations(models_dict, X_test, y_test, class_names,
                          n_samples=50, method="saliency"):
    """
    Compare saliency maps across precision variants of the same model.

    Args:
        models_dict: dict mapping precision_tag → model instance
            e.g. {"fp32": model_fp32, "fp16": model_fp16, "int8": model_int8}
        X_test: test data (N, 1, 360)
        y_test: test labels
        class_names: list of class names
        n_samples: number of samples to compare
        method: "saliency" or "integrated_gradients"

    Returns:
        comparison: dict with pairwise similarity metrics
    """
    compute_fn = (compute_saliency if method == "saliency"
                  else compute_integrated_gradients)

    precisions = list(models_dict.keys())
    sample_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)),
                                   replace=False)

    # Compute explanations for all precision levels
    explanations = {}
    for prec, model in models_dict.items():
        device = next(model.parameters()).device
        exps = []
        for idx in sample_idx:
            x = torch.FloatTensor(X_test[idx:idx+1]).to(device)
            try:
                exp = compute_fn(model, x)
                exps.append(exp.flatten())
            except Exception:
                exps.append(np.zeros(X_test.shape[-1]))
        explanations[prec] = np.array(exps)

    # Pairwise comparison
    comparison = {}
    baseline = precisions[0]  # Typically FP32

    for prec in precisions[1:]:
        cos_sims = []
        spear_corrs = []

        for i in range(len(sample_idx)):
            cs = cosine_similarity(explanations[baseline][i],
                                    explanations[prec][i])
            sr = spearman_correlation(explanations[baseline][i],
                                      explanations[prec][i])
            cos_sims.append(cs)
            spear_corrs.append(sr)

        comparison[f"{baseline}_vs_{prec}"] = {
            "cosine_similarity": {
                "mean": float(np.mean(cos_sims)),
                "std": float(np.std(cos_sims)),
                "min": float(np.min(cos_sims)),
                "max": float(np.max(cos_sims)),
            },
            "spearman_correlation": {
                "mean": float(np.mean(spear_corrs)),
                "std": float(np.std(spear_corrs)),
                "min": float(np.min(spear_corrs)),
                "max": float(np.max(spear_corrs)),
            },
        }

    return comparison, explanations, sample_idx


def plot_explanation_drift(comparison, method_name="Saliency",
                            save_path=None):
    """Plot explanation similarity across precision levels."""
    pairs = list(comparison.keys())
    cos_means = [comparison[p]["cosine_similarity"]["mean"] for p in pairs]
    cos_stds  = [comparison[p]["cosine_similarity"]["std"] for p in pairs]
    sp_means  = [comparison[p]["spearman_correlation"]["mean"] for p in pairs]
    sp_stds   = [comparison[p]["spearman_correlation"]["std"] for p in pairs]

    x = np.arange(len(pairs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, cos_means, width, yerr=cos_stds,
                   label="Cosine Similarity", color="steelblue", capsize=5)
    bars2 = ax.bar(x + width/2, sp_means, width, yerr=sp_stds,
                   label="Spearman Correlation", color="coral", capsize=5)

    ax.set_ylabel("Score")
    ax.set_title(f"Explanation Consistency ({method_name}) Across Precision Levels")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in pairs], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect match")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_explanation_examples(explanations, X_test, y_test,
                               sample_idx, class_names,
                               save_path=None, n_show=3):
    """
    Plot side-by-side saliency maps across precision levels
    for a few sample beats.
    """
    precisions = list(explanations.keys())
    n_prec = len(precisions)

    fig, axes = plt.subplots(n_show, n_prec + 1, figsize=(4 * (n_prec + 1), 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_show):
        idx = sample_idx[row]
        signal = X_test[idx, 0]
        true_cls = y_test[idx]

        # Plot original signal
        axes[row, 0].plot(signal, color="black", lw=0.8)
        axes[row, 0].set_title(f"Signal ({class_names[true_cls]})")
        axes[row, 0].set_ylabel(f"Sample {row}")

        # Plot saliency for each precision
        for col, prec in enumerate(precisions):
            sal = explanations[prec][row]
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            axes[row, col + 1].imshow(
                sal_norm[np.newaxis, :], aspect="auto", cmap="hot",
                extent=[0, len(signal), 0, 1]
            )
            axes[row, col + 1].set_title(prec.upper())

    plt.suptitle("Saliency Maps Across Precision Levels", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_cross_precision_analysis(models_dict, X_test, y_test,
                                  class_names, figures_dir, metrics_dir,
                                  model_name="model", n_samples=50):
    """Full cross-precision interpretability comparison."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    results = {}

    for method in ["saliency", "integrated_gradients"]:
        print(f"\n  ── Cross-precision comparison: {method} ──")

        comparison, explanations, sample_idx = compare_explanations(
            models_dict, X_test, y_test, class_names,
            n_samples=n_samples, method=method,
        )

        results[method] = comparison

        # Print summary
        for pair, metrics in comparison.items():
            cos = metrics["cosine_similarity"]
            sp  = metrics["spearman_correlation"]
            print(f"    {pair}: cosine={cos['mean']:.4f}±{cos['std']:.4f}, "
                  f"spearman={sp['mean']:.4f}±{sp['std']:.4f}")

        # Plot
        plot_explanation_drift(
            comparison, method_name=method.replace("_", " ").title(),
            save_path=os.path.join(figures_dir,
                f"explanation_drift_{model_name}_{method}.png"),
        )

        plot_explanation_examples(
            explanations, X_test, y_test, sample_idx, class_names,
            save_path=os.path.join(figures_dir,
                f"explanation_examples_{model_name}_{method}.png"),
            n_show=3,
        )

    # Save metrics
    with open(os.path.join(metrics_dir,
              f"cross_precision_{model_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Cross-precision analysis complete for {model_name}")
    return results
