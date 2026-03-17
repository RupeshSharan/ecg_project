# Performance and Interpretability Analysis of Mixed Precision and Quantized ECG Classification Models

A comprehensive ML research pipeline for analysing how **mixed precision** (FP16/BF16) and **quantization** (INT8/INT4) affect both the **performance** and **interpretability** of ECG classification models on the MIT-BIH Arrhythmia Database.

## Problem Identification

Arrhythmia classification from ECG signals is critical for early diagnosis of cardiovascular diseases. However, deploying highly accurate deep learning models for continuous ECG monitoring on edge devices or wearable tech is challenging due to strict constraints on power, memory, and computational capacity. The problem lies in balancing classification performance with efficiency (via quantization and mixed precision) without sacrificing the interpretability and reliability that medical settings demand.

## Motivation

As wearable ECG monitors become ubiquitous, there is a pressing need for models that can operate efficiently on-device (resource-constrained devices) while maintaining clinical-grade accuracy. Furthermore, in the healthcare domain, "black box" decisions are unacceptable; cardiologists must be able to trust and verify model predictions. This motivates a comprehensive study of how model compression techniques like mixed precision and quantization impact not just latency and size, but also the actual behavior and interpretability (e.g., saliency maps, SHAP values) of various neural network architectures.

## Background Study

Electrocardiogram (ECG) beat classification typically involves analyzing time-series data to detect abnormal heart rhythms (arrhythmias). Deep learning approaches, particularly 1D Convolutional Neural Networks (CNNs), ResNets, BiLSTMs, and Transformers, have shown state-of-the-art results on datasets like the MIT-BIH Arrhythmia Database. However, research into the trade-offs of deploying these models using lower precision formats (FP16, BF16) and INT8 quantization is still evolving. Studies show that while quantization reduces model size and latency, it can sporadically alter the decision boundaries and internal attention mechanisms. This project systematically analyzes these effects by combining mixed-precision training, Post-Training Quantization (PTQ), and Quantization-Aware Training (QAT) with various interpretability methods like Grad-CAM, SHAP, and LIME.

## Mixed Precision and Quantization in Our Project

In deep learning, reducing the precision of the numerical formats used to represent model weights and activations is a key strategy for optimizing models for edge devices. This project specifically focuses on two main techniques:

**Mixed Precision Training (FP16 / BF16)**
Mixed precision involves training models using both lower-precision 16-bit and standard 32-bit floating-point types to make them execute faster and consume less memory, while aiming to keep the same accuracy as a full 32-bit (FP32) model. 
- **In our project:** We utilize PyTorch's Automatic Mixed Precision (`torch.cuda.amp`) engine (`src/training/train_mixed_precision.py`) to train our baseline models in FP16 and BF16 formats. We analyze how this impacts both the training trajectory and memory consumption without substantially penalizing baseline accuracy.

**Quantization (INT8)**
Quantization goes a step further by mapping continuous floating-point values to discrete integer values (e.g., INT8). This drastically reduces the model's physical footprint and inference latency, optimizing it for the hardware constraints typical of wearable tech.
- **In our project:** We explore multiple INT8 quantization scenarios using `torch.quantization`. Our pipeline evaluates:
  1. **Post-Training Quantization (PTQ):** We apply both dynamic and static INT8 calibration to our fully trained models (`src/quantization/ptq.py`) to analyze the direct edge-deployment trade-off in accuracy.
  2. **Quantization-Aware Training (QAT):** We fine-tune the models while simulating quantization during the forward pass (`src/quantization/qat.py`) to recover performance drops.
  
Finally, we export these mathematically compressed models to ONNX to perform rigorous CPU latency benchmarks (`src/quantization/export_models.py`) and, crucially, measure interpretability drift—analyzing how the "reasoning" of the model and SHAP/Grad-CAM outputs shift due to precision loss.

---

## 📁 Project Structure

```
ecg_project/
├── configs/
│   └── config.yaml              ← Central configuration
├── data/
│   ├── raw/                     ← MIT-BIH WFDB files (.dat/.hea/.atr)
│   ├── processed/               ← Full processed arrays
│   └── splits/                  ← Train/val/test .npy splits
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py        ← Bandpass filter → segment → normalise → split
├── models/
│   │   ├── cnn1d.py             ← 3-block 1D-CNN
│   │   ├── resnet1d.py          ← 1D ResNet with skip connections
│   │   ├── bilstm.py            ← Bidirectional LSTM
│   │   └── transformer1d.py     ← 1D Transformer with CLS token
│   ├── training/
│   │   ├── trainer.py           ← Unified training engine (FP32/FP16/BF16)
│   │   ├── train_baselines.py   ← Train all 4 models in FP32
│   │   └── train_mixed_precision.py ← FP16/BF16 AMP training
│   ├── quantization/
│   │   ├── ptq.py               ← Post-Training Quantization (INT8 static/dynamic)
│   │   ├── qat.py               ← Quantization-Aware Training
│   │   └── export_models.py     ← ONNX export + CPU benchmark
│   ├── interpretability/
│   │   ├── gradcam1d.py         ← Grad-CAM for 1D CNNs
│   │   ├── saliency.py          ← Saliency / Integrated Gradients / SmoothGrad
│   │   ├── shap_analysis.py     ← DeepSHAP / KernelSHAP
│   │   ├── lime_analysis.py     ← LIME for ECG segments
│   │   ├── attention_viz.py     ← Transformer attention weights
│   │   └── cross_precision.py   ← Explanation drift across precisions
│   └── evaluation/
│       ├── evaluator.py         ← Metrics, confusion matrix, ROC
│       ├── comparative_analysis.py ← Master comparison table + Pareto plots
│       └── ablation.py          ← Quantization granularity, calibration, faithfulness
├── notebooks/
│   └── 01_eda.py                ← Exploratory data analysis
├── results/
│   ├── figures/                 ← All generated plots
│   ├── metrics/                 ← JSON metrics + CSV tables
│   ├── checkpoints/             ← Model weights (.pt)
│   ├── onnx_models/             ← ONNX exported models
│   └── tflite_models/           ← TFLite converted models
├── run_all.py                   ← Master orchestration script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/Mac

# Install PyTorch with CUDA (if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### 2. Prepare data

Copy your MIT-BIH WFDB files (`.dat`, `.hea`, `.atr`) into `data/raw/`:

```bash
copy ..\mit-bih-arrhythmia-database-1.0.0\*.dat data\raw\
copy ..\mit-bih-arrhythmia-database-1.0.0\*.hea data\raw\
copy ..\mit-bih-arrhythmia-database-1.0.0\*.atr data\raw\
```

### 3. Run the pipeline

```bash
# Full pipeline (all phases)
python run_all.py

# Or run individual phases
python run_all.py --phase 1          # Preprocessing only
python run_all.py --phase 2          # Train baselines (FP32)
python run_all.py --phase 3          # Mixed precision training
python run_all.py --phase 4          # Quantization (PTQ + QAT + ONNX)
python run_all.py --phase 5          # Interpretability analysis
python run_all.py --phase 6          # Comparative analysis
python run_all.py --phase 7          # Ablation studies
```

---

## 📊 Models

| Architecture | Parameters | Description |
|---|---|---|
| **1D-CNN** | ~100K | 3-block Conv1D → BN → ReLU → MaxPool + FC head |
| **ResNet1D** | ~800K | Deep residual network with skip connections |
| **BiLSTM** | ~400K | Bidi LSTM with conv front-end |
| **Transformer1D** | ~200K | Patch embedding + CLS token + multi-head attention |

---

## 📥 Model Input

The input to the models consists of segmented individual heartbeats (ECG signals) processed from the raw MIT-BIH dataset.

Specifically, the **Input given to the model** undergoes the following pipeline:
1. **Filtering:** Bandpass filtering (0.5 to 40 Hz) is applied to remove noise and baseline wander.
2. **Segmentation:** The continuous ECG signal is segmented around the R-peaks.
3. **Sizing:** Each extracted beat segment is exactly **360 samples** long.
4. **Format:** The models accept a 1D sequence of shape `(Batch_Size, 1, 360)`. 
These 360 numerical values represent the amplitude of the ECG signal over time for a single heartbeat. The models map this input sequence to one of the 5 AAMI classification categories.

---

## 🔬 Precision Formats

| Format | Method | Description |
|---|---|---|
| **FP32** | Baseline | Full 32-bit floating point |
| **FP16** | `torch.cuda.amp` | Automatic Mixed Precision (16-bit forward) |
| **BF16** | `torch.cuda.amp` | Brain Floating Point (requires Ampere+ GPU) |
| **INT8 Dynamic** | `torch.quantization` | Dynamic weight quantization |
| **INT8 Static** | `torch.quantization` | Static quantization with calibration |
| **QAT INT8** | `torch.quantization` | Quantization-Aware Training |

---

## 🧠 Interpretability Methods

| Method | Library | Type |
|---|---|---|
| **Grad-CAM** | Custom 1D | Gradient-based (conv layers) |
| **Saliency Maps** | Captum | Gradient-based (input) |
| **Integrated Gradients** | Captum | Gradient-based (path integral) |
| **SmoothGrad** | Captum | Gradient-based (noise-averaged) |
| **DeepSHAP** | SHAP | Shapley values |
| **LIME** | LIME | Perturbation-based |
| **Attention Viz** | Custom | Transformer attention weights |

---

## 📈 Results

After running the pipeline, find results in:

- **`results/figures/`** — Training curves, confusion matrices, saliency maps, Pareto plots
- **`results/metrics/`** — JSON metrics, master comparison CSV
- **`results/checkpoints/`** — Model weights for all precision variants

---

## 📋 Dataset

**MIT-BIH Arrhythmia Database** — 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects, with ~110,000 annotated beats mapped to 5 AAMI classes:

| Class | Label | Description |
|---|---|---|
| 0 | Normal (N) | Normal beat + bundle branch blocks |
| 1 | SVEB (S) | Supraventricular ectopic beats |
| 2 | VEB (V) | Ventricular ectopic beats |
| 3 | Fusion (F) | Fusion of normal and ventricular |
| 4 | Paced (Q) | Paced beats + unclassifiable |

---

## 🛠 Tech Stack

- **PyTorch 2.x** — Models, training, quantization
- **Captum** — Gradient-based interpretability
- **SHAP** — Shapley value explanations
- **LIME** — Perturbation-based explanations
- **ONNX Runtime** — Inference benchmarking
- **Matplotlib / Seaborn** — Visualisation
- **wfdb** — ECG data loading

---

## Interactive Inference UI

Use the Streamlit frontend to run live inference with trained checkpoints.

### Local run

```bash
python -m streamlit run streamlit_dashboard.py
```

### Features

- Select model checkpoint (`cnn1d`, `resnet1d`, `bilstm`, `transformer`)
- Input ECG signal from:
  - test split sample
  - uploaded `.npy` / `.csv` / `.txt`
  - manual comma-separated values
- Run prediction and view:
  - predicted class + confidence
  - class probability chart
  - gradient saliency curve
- Browse metrics in a separate tab

---

## Docker

### Build

```bash
docker build -t ecg-ui:latest .
```

### Run

```bash
docker run --rm -p 8501:8501 ecg-ui:latest
```

Or with compose:

```bash
docker compose up --build
```

---

## Kubernetes

1. Push your image to a registry, then update image name in:
   - `k8s/deployment.yaml`
2. Apply manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

3. Access with port-forward:

```bash
kubectl port-forward service/ecg-ui 8501:80
```
