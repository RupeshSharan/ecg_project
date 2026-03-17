"""
MIT-BIH Arrhythmia preprocessing pipeline.

Loads WFDB records, applies bandpass filtering, segments beats around
R-peaks, normalizes each beat, maps to AAMI 5 classes, and saves
train/val/test splits as .npy files.
"""

import os
import sys
from collections import Counter

import numpy as np
import wfdb
import yaml
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_PATH = os.path.join(PROJECT_ROOT, cfg["data"]["raw_path"])
PROC_PATH = os.path.join(PROJECT_ROOT, cfg["data"]["processed_path"])
SPLIT_PATH = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
FS = cfg["data"]["fs"]
WINDOW = cfg["data"]["window"]
AAMI_MAP = cfg["data"]["aami_map"]
RECORDS = [str(r) for r in cfg["data"]["records"]]
CLASS_NAMES = cfg["data"]["class_names"]
SEED = cfg["training"]["seed"]
TEST_SIZE = cfg["data"]["test_size"]
VAL_SIZE = cfg["data"]["val_size"]

os.makedirs(PROC_PATH, exist_ok=True)
os.makedirs(SPLIT_PATH, exist_ok=True)


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def extract_beats(data_path=None):
    """
    Read configured MIT-BIH records and return beat-level arrays.
    Output shapes:
      X: (N, 360), y: (N,)
    """
    if data_path is None:
        data_path = RAW_PATH

    X, y = [], []

    for rec_id in tqdm(RECORDS, desc="Processing records"):
        path = os.path.join(data_path, rec_id)
        try:
            record = wfdb.rdrecord(path)
            annotation = wfdb.rdann(path, "atr")
        except Exception as exc:
            print(f"Skipping {rec_id}: {exc}")
            continue

        signal = record.p_signal[:, 0]
        signal = bandpass_filter(signal, fs=FS)

        for peak, sym in zip(annotation.sample, annotation.symbol):
            if sym not in AAMI_MAP:
                continue

            start = peak - WINDOW
            end = peak + WINDOW
            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)
            X.append(beat)
            y.append(AAMI_MAP[sym])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def main():
    """Run full preprocessing and save processed arrays and splits."""
    print("=" * 60)
    print("  MIT-BIH Preprocessing Pipeline")
    print("=" * 60)

    X, y = extract_beats()
    if X.size == 0:
        raise RuntimeError(
            f"No beats extracted from {RAW_PATH}. "
            "Check dataset files and config paths."
        )

    print(f"\nTotal beats extracted : {len(X):,}")
    print(f"Signal length per beat: {X.shape[1]} samples @ {FS} Hz")

    counts = Counter(y.tolist())
    print("\nClass Distribution:")
    for cls, name in enumerate(CLASS_NAMES):
        n = counts.get(cls, 0)
        pct = (n / len(y)) * 100.0
        print(f"Class {cls} - {name:20s}: {n:6,} ({pct:.1f}%)")

    np.save(os.path.join(PROC_PATH, "X_all.npy"), X)
    np.save(os.path.join(PROC_PATH, "y_all.npy"), y)

    # Conv1D format: (N, C, L)
    X = X[:, np.newaxis, :]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE, stratify=y_tmp, random_state=SEED
    )

    print("\nSplit sizes:")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    for name, arr in [
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
        ("X_test", X_test),
        ("y_test", y_test),
    ]:
        np.save(os.path.join(SPLIT_PATH, f"{name}.npy"), arr)

    print(f"\nAll splits saved to {SPLIT_PATH}{os.sep}")
    print("Next step: run src/training/train_baselines.py")


if __name__ == "__main__":
    main()
