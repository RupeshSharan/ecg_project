"""
EDA Script for MIT-BIH Arrhythmia Database
============================================
Run in VS Code interactive window (# %% cell markers) or as a notebook.
Generates ECG signal plots, annotated beats, and class distribution charts.
"""

# %%
import os
import sys
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw")
SPLITS_DIR  = os.path.join(PROJECT_ROOT, "data", "splits")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# %% ── 1. Load a single record ───────────────────────────────────────
record     = wfdb.rdrecord(os.path.join(RAW_DIR, "100"))
annotation = wfdb.rdann(os.path.join(RAW_DIR, "100"), "atr")

fs     = record.fs
signal = record.p_signal[:, 0]   # Lead MLII
peaks  = annotation.sample
syms   = annotation.symbol

print(f"Record 100 — shape: {record.p_signal.shape}, fs: {fs} Hz")
print(f"Leads: {record.sig_name}")
print(f"Unique annotation symbols: {sorted(set(syms))}")

# %% ── 2. Raw ECG strip + annotated beats ────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# 2a — Raw 10-second strip
t = np.arange(10 * fs) / fs
axes[0].plot(t, signal[:10 * fs], lw=0.8, color="royalblue")
axes[0].set_title("Record 100 — Lead MLII (first 10 seconds)", fontsize=13)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude (mV)")

# 2b — Beat annotations overlaid
axes[1].plot(signal[:10 * fs], lw=0.6, color="gray")
for p, s in zip(peaks, syms):
    if p < 10 * fs:
        axes[1].axvline(p, color="red", alpha=0.5, lw=0.8)
        axes[1].text(p, signal[p] + 0.15, s, fontsize=7, color="red", ha="center")
axes[1].set_title("Beat Annotations Overlaid")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Amplitude (mV)")

# 2c — Sample individual beats
W = 180
beat_count = 0
colours = plt.cm.tab10.colors
for p, s in zip(peaks, syms):
    if s == "N" and W < p < len(signal) - W and beat_count < 6:
        axes[2].plot(
            np.arange(-W, W),
            signal[p - W : p + W],
            alpha=0.7,
            lw=1.0,
            color=colours[beat_count],
            label=f"Beat @{p}",
        )
        beat_count += 1
axes[2].set_title("Sample Normal Beats (centred on R-peak)")
axes[2].set_xlabel("Sample offset from R-peak")
axes[2].set_ylabel("Amplitude (mV)")
axes[2].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_record100.png"), dpi=150)
plt.show()
print("✅ Saved eda_record100.png")

# %% ── 3. Sample beats from different classes ────────────────────────
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}
class_names = ["Normal (N)", "SVEB (S)", "VEB (V)", "Fusion (F)", "Paced (/Q)"]

# Collect one sample beat per class from record 200 (has more variety)
rec200 = wfdb.rdrecord(os.path.join(RAW_DIR, "200"))
ann200 = wfdb.rdann(os.path.join(RAW_DIR, "200"), "atr")
sig200 = rec200.p_signal[:, 0]

class_beats = {}
for p, s in zip(ann200.sample, ann200.symbol):
    if s in AAMI_MAP:
        cls = AAMI_MAP[s]
        if cls not in class_beats and W < p < len(sig200) - W:
            class_beats[cls] = sig200[p - W : p + W]
    if len(class_beats) == 5:
        break

fig, axes = plt.subplots(1, len(class_beats), figsize=(18, 3), sharey=True)
for cls in sorted(class_beats.keys()):
    ax = axes[cls]
    ax.plot(class_beats[cls], lw=0.9, color=colours[cls])
    ax.set_title(class_names[cls], fontsize=11)
    ax.set_xlabel("Sample")
    if cls == 0:
        ax.set_ylabel("Amplitude")
plt.suptitle("Sample Beat per AAMI Class (Record 200)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_class_samples.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved eda_class_samples.png")

# %% ── 4. Class distribution across splits ───────────────────────────
# Run AFTER preprocessing has been completed
try:
    y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

    class_short = ["Normal", "SVEB", "VEB", "Fusion", "Paced"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (split_y, title) in zip(axes, [
        (y_train, "Train"), (y_val, "Validation"), (y_test, "Test"),
    ]):
        cnts = [Counter(split_y).get(i, 0) for i in range(5)]
        bars = ax.bar(class_short, cnts, color="steelblue", edgecolor="black")
        ax.set_title(f"{title} Set ({len(split_y):,} beats)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
        for bar, c in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{c:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.show()
    print("✅ Saved class_distribution.png")
except FileNotFoundError:
    print("⚠ Splits not found — run src/preprocessing/preprocess.py first.")

# %% ── 5. Signal statistics ──────────────────────────────────────────
try:
    X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
    print(f"\nBeat shape (train): {X_train.shape}")
    print(f"  dtype      : {X_train.dtype}")
    print(f"  mean       : {X_train.mean():.4f}")
    print(f"  std        : {X_train.std():.4f}")
    print(f"  min / max  : {X_train.min():.4f} / {X_train.max():.4f}")
    print(f"  beat length: {X_train.shape[2]} samples = {X_train.shape[2]/360:.2f} s")
except FileNotFoundError:
    pass
