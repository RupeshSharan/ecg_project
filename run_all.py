"""
Master pipeline runner for ECG quantization + interpretability project.

Usage:
  python run_all.py
  python run_all.py --phase 1
  python run_all.py --phase 2 3 4
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Avoid Windows cp1252 crashes when modules print Unicode symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_config():
    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def phase1_preprocessing():
    """Phase 1: Preprocess MIT-BIH data."""
    print("\n" + "#" * 60)
    print("PHASE 1 - PREPROCESSING")
    print("#" * 60)

    cfg = load_config()
    splits_path = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
    train_split = os.path.join(splits_path, "X_train.npy")

    if os.path.exists(train_split):
        print("Splits already exist. Skipping preprocessing.")
        return

    from src.preprocessing.preprocess import main as preprocess_main

    preprocess_main()


def phase2_baselines():
    """Phase 2: Train all baseline models (FP32)."""
    print("\n" + "#" * 60)
    print("PHASE 2 - BASELINE MODELS (FP32)")
    print("#" * 60)
    from src.training.train_baselines import main as train_baselines_main

    train_baselines_main()


def phase3_mixed_precision():
    """Phase 3: Mixed precision training (FP16, BF16)."""
    print("\n" + "#" * 60)
    print("PHASE 3 - MIXED PRECISION TRAINING")
    print("#" * 60)
    from src.training.train_mixed_precision import main as train_mixed_main

    train_mixed_main()


def phase4_quantization():
    """Phase 4: PTQ, QAT, and export."""
    print("\n" + "#" * 60)
    print("PHASE 4 - QUANTIZATION")
    print("#" * 60)

    print("\n-- PTQ --")
    from src.quantization.ptq import main as ptq_main

    ptq_main()

    print("\n-- QAT --")
    from src.quantization.qat import main as qat_main

    qat_main()

    print("\n-- ONNX Export & Benchmark --")
    from src.quantization.export_models import main as export_main

    export_main()


def phase5_interpretability():
    """Phase 5: Run interpretability analyses."""
    print("\n" + "#" * 60)
    print("PHASE 5 - INTERPRETABILITY")
    print("#" * 60)

    cfg = load_config()
    splits = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
    fig_dir = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])

    X_test = np.load(os.path.join(splits, "X_test.npy"))
    y_test = np.load(os.path.join(splits, "y_test.npy"))
    X_train = np.load(os.path.join(splits, "X_train.npy"))

    from src.models.cnn1d import CNN1D
    from src.models.transformer1d import Transformer1D
    from src.quantization.ptq import load_pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = load_pretrained(CNN1D, "cnn1d", device, input_length=360, num_classes=5)
    if cnn is not None:
        cnn = cnn.to(device)

        from src.interpretability.gradcam1d import run_gradcam_analysis
        from src.interpretability.lime_analysis import run_lime_analysis
        from src.interpretability.saliency import run_saliency_analysis
        from src.interpretability.shap_analysis import run_shap_analysis

        target_layer = cnn.features[5]
        run_gradcam_analysis(
            cnn,
            X_test,
            y_test,
            target_layer,
            cfg["data"]["class_names"],
            fig_dir,
            "cnn1d",
            "fp32",
        )
        run_saliency_analysis(
            cnn, X_test, y_test, cfg["data"]["class_names"], fig_dir, "cnn1d", "fp32"
        )
        run_shap_analysis(
            cnn,
            X_test,
            y_test,
            X_train,
            cfg["data"]["class_names"],
            fig_dir,
            "cnn1d",
            "fp32",
        )
        run_lime_analysis(
            cnn,
            X_test,
            y_test,
            X_train,
            cfg["data"]["class_names"],
            fig_dir,
            "cnn1d",
            "fp32",
        )

    trans = load_pretrained(Transformer1D, "transformer", device, input_length=360, num_classes=5)
    if trans is not None:
        trans = trans.to(device)
        from src.interpretability.attention_viz import run_attention_analysis

        run_attention_analysis(
            trans, X_test, y_test, cfg["data"]["class_names"], fig_dir, "transformer", "fp32"
        )


def phase6_analysis():
    """Phase 6: Comparative analysis."""
    print("\n" + "#" * 60)
    print("PHASE 6 - COMPARATIVE ANALYSIS")
    print("#" * 60)
    cfg = load_config()
    from src.evaluation.comparative_analysis import run_comparative_analysis

    run_comparative_analysis(
        os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"]),
        os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"]),
    )


def phase7_ablation():
    """Phase 7: Ablation studies."""
    print("\n" + "#" * 60)
    print("PHASE 7 - ABLATION STUDIES")
    print("#" * 60)

    cfg = load_config()
    splits = os.path.join(PROJECT_ROOT, cfg["data"]["splits_path"])
    fig_dir = os.path.join(PROJECT_ROOT, cfg["results"]["figures_dir"])
    met_dir = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])

    X_train = np.load(os.path.join(splits, "X_train.npy"))
    X_test = np.load(os.path.join(splits, "X_test.npy"))
    y_test = np.load(os.path.join(splits, "y_test.npy"))

    from src.evaluation.ablation import (
        calibration_size_sweep,
        compare_quantization_granularity,
        run_faithfulness_analysis,
    )
    from src.models.cnn1d import CNN1D
    from src.quantization.ptq import load_pretrained

    model = load_pretrained(CNN1D, "cnn1d", "cpu", input_length=360, num_classes=5)
    if model is None:
        print("No trained model found. Skipping ablation.")
        return

    np.random.seed(42)
    cal_size = min(1000, len(X_train))
    X_cal = X_train[np.random.choice(len(X_train), cal_size, replace=False)]

    print("\n-- 7.1 Quantization Granularity --")
    compare_quantization_granularity(
        model, X_cal, X_test, y_test, cfg["data"]["class_names"], fig_dir, met_dir
    )

    print("\n-- 7.2 Calibration Data Size Sweep --")
    calibration_size_sweep(model, X_train, X_test, y_test, cfg["data"]["class_names"], fig_dir, met_dir)

    print("\n-- 7.3 Faithfulness Tests --")
    run_faithfulness_analysis(
        model, X_test, y_test, cfg["data"]["class_names"], fig_dir, met_dir, "cnn1d", "fp32"
    )


PHASES = {
    1: ("Preprocessing", phase1_preprocessing),
    2: ("Baselines (FP32)", phase2_baselines),
    3: ("Mixed Precision", phase3_mixed_precision),
    4: ("Quantization", phase4_quantization),
    5: ("Interpretability", phase5_interpretability),
    6: ("Analysis", phase6_analysis),
    7: ("Ablation", phase7_ablation),
}


def main():
    parser = argparse.ArgumentParser(description="ECG Quantization Pipeline")
    parser.add_argument(
        "--phase",
        nargs="+",
        type=int,
        default=list(PHASES.keys()),
        help="Phase(s) to run. Example: --phase 2 3",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ECG Quantization & Interpretability Pipeline")
    print("=" * 60)

    for phase_id in sorted(args.phase):
        if phase_id not in PHASES:
            print(f"Unknown phase: {phase_id}")
            continue

        phase_name, fn = PHASES[phase_id]
        print(f"\nStarting Phase {phase_id}: {phase_name}")
        fn()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
