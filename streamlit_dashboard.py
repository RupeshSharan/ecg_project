"""
Interactive Streamlit frontend for ECG model inference + metrics.

Run:
  streamlit run streamlit_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

from src.models.bilstm import BiLSTM
from src.models.cnn1d import CNN1D
from src.models.resnet1d import ResNet1D
from src.models.transformer1d import Transformer1D


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


MODEL_REGISTRY = {
    "cnn1d": (CNN1D, {"input_length": 360, "num_classes": 5}),
    "resnet1d": (ResNet1D, {"input_length": 360, "num_classes": 5}),
    "bilstm": (BiLSTM, {"input_length": 360, "num_classes": 5}),
    "transformer": (Transformer1D, {"input_length": 360, "num_classes": 5}),
}


def load_config() -> Dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


CFG = load_config()
CLASS_NAMES = CFG.get("data", {}).get("class_names", ["Normal", "SVEB", "VEB", "Fusion", "Paced"])
CHECKPOINT_DIR = PROJECT_ROOT / CFG.get("models", {}).get("checkpoint_dir", "results/checkpoints")
SPLITS_DIR = PROJECT_ROOT / CFG.get("data", {}).get("splits_path", "data/splits")
METRICS_DIR = PROJECT_ROOT / CFG.get("results", {}).get("metrics_dir", "results/metrics")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* Main page background */
        .stApp {
            background: #FFFFFF !important;
        }
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #87CEEB !important;
        }
        /* Sidebar text styling */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            font-size: 1.65rem !important;
            font-weight: 900 !important;
            color: #0f3b57 !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
            font-size: 1.25rem !important;
            font-weight: normal !important;
            color: #154360 !important;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #154360 !important;
        }
        /* Align main title with sidebar */
        .block-container {
            padding-top: 2.2rem !important;
        }
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        html, body, [class*="css"] {
            font-family: "Trebuchet MS", "Lucida Sans", "Verdana", sans-serif;
        }
        .card {
            background: #ffffff;
            border: 1px solid #dbe2ea;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        .subtle {
            color: #486581;
            font-size: 0.95rem;
        }
        /* Hide "Press Ctrl+Enter" hint */
        [data-testid="InputInstructions"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_metrics(metrics_dir: str) -> pd.DataFrame:
    base = Path(metrics_dir)
    rows: List[Dict] = []

    for path in sorted(base.glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows.append(
            {
                "file": path.name,
                "model": payload.get("model", "unknown"),
                "precision": payload.get("precision", "unknown"),
                "accuracy": float(payload.get("accuracy", 0.0) or 0.0),
                "f1_macro": float(payload.get("f1_macro", 0.0) or 0.0),
                "auc_roc": float(payload.get("auc_roc", 0.0) or 0.0),
                "latency_ms": float(payload.get("latency_ms", {}).get("mean_ms", 0.0) or 0.0),
                "model_size_mb": float(payload.get("model_size_mb", 0.0) or 0.0),
                "total_params": int(payload.get("total_params", 0) or 0),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "file",
                "model",
                "precision",
                "accuracy",
                "f1_macro",
                "auc_roc",
                "latency_ms",
                "model_size_mb",
                "total_params",
            ]
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_test_split(splits_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p = Path(splits_dir)
    x_path = p / "X_test.npy"
    y_path = p / "y_test.npy"
    if not (x_path.exists() and y_path.exists()):
        return None, None
    return np.load(x_path), np.load(y_path)


def discover_checkpoints(checkpoint_dir: Path) -> List[Dict]:
    options: List[Dict] = []
    if not checkpoint_dir.exists():
        return options

    for ckpt in sorted(checkpoint_dir.glob("*_best.pt")):
        stem = ckpt.stem  # e.g., cnn1d_fp32_best
        if not stem.endswith("_best"):
            continue
        base = stem[: -len("_best")]
        model_name = None
        tag = None
        for candidate in MODEL_REGISTRY.keys():
            prefix = f"{candidate}_"
            if base.startswith(prefix):
                model_name = candidate
                tag = base[len(prefix) :]
                break
        if model_name is None or tag is None:
            continue

        options.append(
            {
                "label": f"{model_name} ({tag})",
                "model_name": model_name,
                "tag": tag,
                "path": str(ckpt.resolve()),
            }
        )
    return options


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str, checkpoint_path: str):
    model_cls, kwargs = MODEL_REGISTRY[model_name]
    model = model_cls(**kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32).reshape(-1)
    mean = signal.mean()
    std = signal.std()
    return (signal - mean) / (std + 1e-8)


def coerce_signal_shape(raw: np.ndarray, target_len: int = 360) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32).squeeze()
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.size == target_len:
        return x
    if x.size > target_len:
        return x[:target_len]
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: x.size] = x
    return padded


def parse_uploaded_signal(uploaded_file) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".npy":
        arr = np.load(uploaded_file, allow_pickle=False)
        return coerce_signal_shape(arr)

    if suffix in {".csv", ".txt"}:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            arr = df.values
        except Exception:
            uploaded_file.seek(0)
            arr = np.loadtxt(uploaded_file, delimiter=",")
        return coerce_signal_shape(arr)

    raise ValueError(f"Unsupported file type: {suffix}")


def predict_signal(model, signal_360: np.ndarray) -> Tuple[np.ndarray, int]:
    x = torch.from_numpy(signal_360.astype(np.float32)).view(1, 1, -1)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx


def compute_saliency_1d(model, signal_360: np.ndarray, target_idx: int) -> np.ndarray:
    x = torch.from_numpy(signal_360.astype(np.float32)).view(1, 1, -1)
    x.requires_grad_(True)
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    model.zero_grad(set_to_none=True)
    out[0, target_idx].backward()
    sal = x.grad.detach().abs().cpu().numpy().reshape(-1)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal


def render_inference_tab() -> None:
    st.markdown('<div class="card"><h3>Live Model Inference</h3><p class="subtle">Select a trained checkpoint, provide an ECG beat signal, and run prediction.</p></div>', unsafe_allow_html=True)

    ckpt_options = discover_checkpoints(CHECKPOINT_DIR)
    if not ckpt_options:
        st.error(f"No checkpoints found in: {CHECKPOINT_DIR}")
        return

    selected_label = st.selectbox("Model checkpoint", [o["label"] for o in ckpt_options], index=0)
    selected = next(o for o in ckpt_options if o["label"] == selected_label)
    model = load_trained_model(selected["model_name"], selected["path"])

    source = st.radio(
        "Input source",
        options=["Use test split sample", "Upload signal file", "Manual values"],
        horizontal=True,
    )
    use_zscore = st.checkbox("Apply Z-score normalization before inference", value=True)
    top_k = st.slider("Top-k probabilities", min_value=1, max_value=len(CLASS_NAMES), value=3, step=1)

    signal: Optional[np.ndarray] = None
    true_label = None

    if source == "Use test split sample":
        X_test, y_test = load_test_split(str(SPLITS_DIR))
        if X_test is None:
            st.warning("Test split not found. Upload a file or use manual values.")
        else:
            idx = st.slider("Test sample index", min_value=0, max_value=int(len(X_test) - 1), value=0, step=1)
            signal = X_test[idx, 0].astype(np.float32)
            true_idx = int(y_test[idx])
            if 0 <= true_idx < len(CLASS_NAMES):
                true_label = CLASS_NAMES[true_idx]

    elif source == "Upload signal file":
        uploaded = st.file_uploader("Upload .npy, .csv, or .txt", type=["npy", "csv", "txt"])
        if uploaded is not None:
            try:
                signal = parse_uploaded_signal(uploaded)
            except Exception as exc:
                st.error(f"Could not parse uploaded file: {exc}")

    else:
        raw = st.text_area(
            "Enter comma-separated values",
            value="",
            placeholder="0.02, -0.10, 0.18, ... (up to 360 values)",
            height=120,
        )
        st.button("Analyze Custom Values ➡️", type="primary")
        
        if raw.strip():
            try:
                vals = np.array([float(v.strip()) for v in raw.split(",") if v.strip()], dtype=np.float32)
                signal = coerce_signal_shape(vals)
            except Exception as exc:
                st.error(f"Invalid numeric input: {exc}")

    if signal is None:
        st.info("Provide an input signal to run prediction.")
        return

    if use_zscore:
        signal = normalize_signal(signal)

    st.line_chart(pd.DataFrame({"ecg_signal": signal}))

    probs, pred_idx = predict_signal(model, signal)
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    confidence = float(probs[pred_idx])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Class", pred_label)
    c2.metric("Confidence", f"{confidence:.4f}")
    c3.metric("Input Length", f"{len(signal)}")
    if true_label is not None:
        st.caption(f"Ground truth label (from test split): {true_label}")

    prob_df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
    prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("class:N", sort=None, axis=alt.Axis(labelAngle=0, title="Class")),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.Color("class:N", legend=None),
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(prob_df.head(top_k), use_container_width=True, hide_index=True)

    with st.expander("Show saliency map (gradient-based)"):
        try:
            saliency = compute_saliency_1d(model, signal, pred_idx)
            st.line_chart(pd.DataFrame({"saliency": saliency}))
        except Exception as exc:
            st.warning(f"Could not compute saliency: {exc}")


def render_metrics_tab() -> None:
    st.markdown('<div class="card"><h3>Metrics Explorer</h3><p class="subtle">Browse accuracy, F1, AUC, latency, and model size from metrics JSON files.</p></div>', unsafe_allow_html=True)

    df = load_metrics(str(METRICS_DIR))
    if df.empty:
        st.warning("No *_metrics.json files found.")
        return

    with st.sidebar:
        st.header("Metrics Filters")
        models = sorted(df["model"].dropna().unique().tolist())
        precisions = sorted(df["precision"].dropna().unique().tolist())
        selected_models = st.multiselect("Model", options=models, default=models)
        selected_precisions = st.multiselect("Precision", options=precisions, default=precisions)
        sort_col = st.selectbox(
            "Sort by",
            options=["accuracy", "f1_macro", "auc_roc", "latency_ms", "model_size_mb"],
            index=0,
        )
        ascending = st.checkbox("Ascending", value=False)

    filtered = df[df["model"].isin(selected_models) & df["precision"].isin(selected_precisions)].copy()
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(filtered))
    c2.metric("Best Accuracy", f"{filtered['accuracy'].max():.4f}")
    c3.metric("Best F1 Macro", f"{filtered['f1_macro'].max():.4f}")
    c4.metric("Fastest Latency", f"{filtered['latency_ms'].min():.2f} ms")

    st.dataframe(
        filtered[
            [
                "model",
                "precision",
                "accuracy",
                "f1_macro",
                "auc_roc",
                "latency_ms",
                "model_size_mb",
                "total_params",
                "file",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Accuracy by run file")
    st.bar_chart(filtered.set_index("file")["accuracy"])

    export_csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered metrics CSV",
        export_csv,
        file_name="filtered_metrics.csv",
        mime="text/csv",
    )


def render_guide_tab() -> None:
    st.markdown('<div class="card"><h3>Welcome to the ECG Quantization Dashboard</h3><p class="subtle">Interactive inference and metrics explorer for the mixed precision and quantized ECG models.</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### 🚀 How to Use This App

    **1. Inference Tab**
    *   **Select a Model:** Choose from any of the trained architectures (CNN1D, ResNet1D, BiLSTM, Transformer) across their various precision levels (FP32, FP16, INT8, QAT).
    *   **Provide a Signal:** 
        *   Choose *Use test split sample* to automatically pick an authentic ECG beat from the processed MIT-BIH test dataset.
        *   *Upload signal file* allows you to input your own `.npy`, `.csv`, or `.txt` signal.
        *   *Manual values* lets you paste a comma-separated array of up to 360 signal values.
    *   **Run Inference:** The app will display the signal, predict the Arrhythmia class, show the confidence levels, and compute a Saliency Map to highlight which parts of the heartbeat influenced the model's decision.

    **2. Metrics Tab**
    *   **Explore Performance:** Browse the test accuracy, F1-scores, AUC, inference latency (ms), and model size (MB) for every model variant you trained.
    *   **Filter & Sort:** Use the sidebar to filter the data by specific architectures or precisions. 
    *   **Export:** Download the filtered metrics table as a CSV for reporting.
    """)


def main() -> None:
    st.set_page_config(page_title="ECG Inference UI", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    
    with st.sidebar:
        st.title("ECG Settings")
        st.info("Use the main tabs to switch between Inference and Metrics.")

    st.title("ECG Arrhythmia Intelligence Dashboard")
    st.caption("Run predictions on ECG beats and inspect model performance metrics.")

    tab_infer, tab_metrics = st.tabs(["Inference", "Metrics"])
    with tab_infer:
        render_inference_tab()
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
