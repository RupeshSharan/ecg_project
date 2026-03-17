# ECG Project: Logic and Usage Guide

## 1) What this project does

This project trains and compares ECG beat classification models on the MIT-BIH Arrhythmia dataset, then analyzes:

- baseline FP32 performance
- mixed-precision behavior (FP16/BF16)
- quantization tradeoffs (INT8 PTQ/QAT)
- interpretability outputs (saliency, Grad-CAM, SHAP, LIME, attention)

It also includes a Streamlit dashboard for interactive inference and metrics exploration.

## 2) Core pipeline logic

The main orchestrator is `run_all.py`, which runs these phases:

1. `Phase 1 - Preprocessing`
2. `Phase 2 - Baselines (FP32)`
3. `Phase 3 - Mixed Precision`
4. `Phase 4 - Quantization`
5. `Phase 5 - Interpretability`
6. `Phase 6 - Comparative Analysis`
7. `Phase 7 - Ablation`

### Phase details

1. **Preprocessing (`src/preprocessing/preprocess.py`)**
- reads MIT-BIH WFDB files from `data/raw/`
- applies bandpass filtering (0.5 to 40 Hz)
- segments each beat around R-peaks (360 samples per beat)
- maps original beat symbols to 5 AAMI classes
- creates train/validation/test numpy splits in `data/splits/`

2. **Baseline training (`src/training/train_baselines.py`)**
- trains `cnn1d`, `resnet1d`, `bilstm`, `transformer` in FP32
- uses common training engine (`src/training/trainer.py`)
- saves best checkpoints and evaluation artifacts

3. **Mixed precision (`src/training/train_mixed_precision.py`)**
- trains the same model set with `fp16` (and `bf16` if CUDA supports it)
- logs epoch timing and memory information

4. **Quantization (`src/quantization/ptq.py`, `src/quantization/qat.py`)**
- PTQ: dynamic and static INT8 quantization where supported
- QAT: fine-tunes quantization-aware model, converts to INT8, evaluates output

5. **Export and benchmark (`src/quantization/export_models.py`)**
- exports trained FP32 models to ONNX
- compares PyTorch CPU latency vs ONNX Runtime latency

6. **Comparative analysis (`src/evaluation/comparative_analysis.py`)**
- loads all `*_metrics.json`
- builds combined comparison table (`master_comparison.csv`)
- generates comparison charts (accuracy/latency/size views)

7. **Ablation (`src/evaluation/ablation.py`)**
- quantization granularity comparison
- calibration-size sweep
- faithfulness tests for explanations

## 3) Important directories

- `configs/config.yaml`: central config for data paths, training, output paths
- `data/raw/`: MIT-BIH WFDB files (`.dat`, `.hea`, `.atr`)
- `data/splits/`: generated `X_train.npy`, `X_val.npy`, `X_test.npy`, etc.
- `results/checkpoints/`: model checkpoint files (`*_best.pt`)
- `results/metrics/`: JSON metrics and summary artifacts
- `results/figures/`: confusion matrices, curves, comparison plots

## 4) How to run locally

From `ecg_project/`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put MIT-BIH files into `data/raw/`, then run:

```bash
python run_all.py
```

Run specific phases only:

```bash
python run_all.py --phase 1
python run_all.py --phase 2 3 4
```

## 5) How to use the Streamlit app

Start app:

```bash
python -m streamlit run streamlit_dashboard.py
```

In the app:

1. Go to **Inference** tab
2. Select a checkpoint from `results/checkpoints/`
3. Choose input source:
- test split sample
- upload `.npy`, `.csv`, or `.txt`
- manual comma-separated values
4. Run prediction and inspect:
- predicted class + confidence
- class probability chart
- saliency curve

Use the **Metrics** tab to filter and compare model runs from `results/metrics/`.

## 6) Docker usage

Build and run:

```bash
docker build -t ecg-ui:latest .
docker run --rm -p 8501:8501 ecg-ui:latest
```

Or:

```bash
docker compose up --build
```

## 7) Expected outputs after full run

- metrics JSON files in `results/metrics/`
- trained checkpoints in `results/checkpoints/`
- figures and plots in `results/figures/`
- ONNX models in `results/onnx_models/` (if export succeeds)
