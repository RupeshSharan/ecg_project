"""
Comparative Performance Analysis
==================================
Build master comparison table, statistical tests, and visualization
of all model variants across precision levels.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def load_all_metrics(metrics_dir):
    """Load all *_metrics.json files into a DataFrame."""
    rows = []
    for f in sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.json"))):
        with open(f) as fp:
            m = json.load(fp)
        rows.append({
            "Model":      m.get("model", ""),
            "Precision":  m.get("precision", ""),
            "Accuracy":   m.get("accuracy", 0),
            "F1_Macro":   m.get("f1_macro", 0),
            "AUC_ROC":    m.get("auc_roc", 0),
            "Latency_ms": m.get("latency_ms", {}).get("mean_ms", 0),
            "Size_MB":    m.get("model_size_mb", 0),
            "Params":     m.get("total_params", 0),
        })
    return pd.DataFrame(rows)


def plot_accuracy_vs_precision(df, save_path):
    """Bar chart comparing accuracy across precision levels per model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    models = df["Model"].unique()
    n_models = len(models)
    precisions = df["Precision"].unique()
    n_prec = len(precisions)

    x = np.arange(n_models)
    width = 0.8 / n_prec
    colors = plt.cm.Set2.colors

    for i, prec in enumerate(precisions):
        subset = df[df["Precision"] == prec]
        vals = [subset[subset["Model"] == m]["Accuracy"].values[0]
                if len(subset[subset["Model"] == m]) > 0 else 0
                for m in models]
        ax.bar(x + i * width, vals, width, label=prec, color=colors[i % len(colors)])

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Across Precision Levels")
    ax.set_xticks(x + width * (n_prec - 1) / 2)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(title="Precision")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pareto_latency_accuracy(df, save_path):
    """Pareto plot: latency vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    models = df["Model"].unique()
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, model in enumerate(models):
        subset = df[df["Model"] == model]
        for _, row in subset.iterrows():
            ax.scatter(row["Latency_ms"], row["Accuracy"],
                       color=colors[i % len(colors)],
                       marker=markers[i % len(markers)],
                       s=100, edgecolors="black", linewidths=0.5)
            ax.annotate(row["Precision"], (row["Latency_ms"], row["Accuracy"]),
                        fontsize=7, ha="center", va="bottom")

    # Add legend for models
    for i, model in enumerate(models):
        ax.scatter([], [], color=colors[i % len(colors)],
                   marker=markers[i % len(markers)], s=100,
                   label=model, edgecolors="black")

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency (Pareto Front)")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pareto_size_accuracy(df, save_path):
    """Pareto plot: model size vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    models = df["Model"].unique()

    for i, model in enumerate(models):
        subset = df[df["Model"] == model]
        ax.scatter(subset["Size_MB"], subset["Accuracy"],
                   color=colors[i % len(colors)], s=100,
                   edgecolors="black", linewidths=0.5, label=model)
        for _, row in subset.iterrows():
            ax.annotate(row["Precision"], (row["Size_MB"], row["Accuracy"]),
                        fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Model Size")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_comparative_analysis(metrics_dir, figures_dir):
    """Run full comparative analysis pipeline."""
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Comparative Performance Analysis")
    print("=" * 60)

    df = load_all_metrics(metrics_dir)

    if df.empty:
        print("  ⚠ No metrics files found. Train models first.")
        return

    # Print master table
    print("\n📊 Master Comparison Table:")
    print(df.to_string(index=False))
    print()

    # Save as CSV
    csv_path = os.path.join(metrics_dir, "master_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved master table → {csv_path}")

    # Generate plots
    plot_accuracy_vs_precision(
        df, os.path.join(figures_dir, "accuracy_vs_precision.png"))
    print("  ✅ Plot: accuracy_vs_precision.png")

    plot_pareto_latency_accuracy(
        df, os.path.join(figures_dir, "pareto_latency_accuracy.png"))
    print("  ✅ Plot: pareto_latency_accuracy.png")

    plot_pareto_size_accuracy(
        df, os.path.join(figures_dir, "pareto_size_accuracy.png"))
    print("  ✅ Plot: pareto_size_accuracy.png")

    return df


if __name__ == "__main__":
    import yaml
    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    run_comparative_analysis(
        os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"]),
        os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"]),
    )
