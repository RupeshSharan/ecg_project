"""
Model Export and Benchmarking
=============================
Export trained models to ONNX and benchmark CPU latency with
ONNX Runtime vs PyTorch.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.bilstm import BiLSTM
from src.models.cnn1d import CNN1D
from src.models.resnet1d import ResNet1D
from src.models.transformer1d import Transformer1D
from src.quantization.ptq import load_pretrained

# Avoid Windows cp1252 crashes when exporter/loggers emit Unicode glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ONNX_DIR = os.path.join(PROJECT_ROOT, cfg["quantization"]["onnx_dir"])
MET_DIR = os.path.join(PROJECT_ROOT, cfg["results"]["metrics_dir"])
ONNX_OPSET = int(cfg["quantization"].get("onnx_opset", 18))
os.makedirs(ONNX_DIR, exist_ok=True)


def export_to_onnx(model, model_name, input_shape=(1, 1, 360)):
    """Export PyTorch model to ONNX."""
    model.eval()
    dummy = torch.randn(*input_shape)
    onnx_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=["ecg_input"],
        output_names=["class_logits"],
        dynamic_axes={"ecg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
    )

    size_mb = os.path.getsize(onnx_path) / (1024**2)
    print(f"  Exported {model_name}.onnx ({size_mb:.2f} MB)")
    return onnx_path


def benchmark_onnx(onnx_path, input_shape=(1, 1, 360), n_runs=200):
    """Benchmark ONNX Runtime CPU inference latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed - skipping ONNX benchmark")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(20):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def benchmark_pytorch(model, input_shape=(1, 1, 360), device="cpu", n_runs=200):
    """Benchmark PyTorch inference latency."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main():
    print(f"Model Export and Benchmarking (ONNX opset {ONNX_OPSET})\n")

    models_config = [
        ("cnn1d", CNN1D, {"input_length": 360, "num_classes": 5}),
        ("resnet1d", ResNet1D, {"input_length": 360, "num_classes": 5}),
        ("bilstm", BiLSTM, {"input_length": 360, "num_classes": 5}),
        ("transformer", Transformer1D, {"input_length": 360, "num_classes": 5}),
    ]

    results = {}

    for name, cls, kwargs in models_config:
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")

        model = load_pretrained(cls, name, "cpu", **kwargs)
        if model is None:
            continue

        pt_lat = benchmark_pytorch(model)
        print(f"  PyTorch CPU: {pt_lat['mean_ms']:.2f} +/- {pt_lat['std_ms']:.2f} ms")

        try:
            onnx_path = export_to_onnx(model, name)
            ort_lat = benchmark_onnx(onnx_path)
            if ort_lat:
                print(f"  ONNX CPU   : {ort_lat['mean_ms']:.2f} +/- {ort_lat['std_ms']:.2f} ms")
                speedup = pt_lat["mean_ms"] / ort_lat["mean_ms"]
                print(f"  Speedup    : {speedup:.2f}x")
        except Exception as exc:
            print(f"  ONNX export failed: {exc}")
            onnx_path = None
            ort_lat = None

        results[name] = {
            "pytorch_cpu_latency": pt_lat,
            "onnx_cpu_latency": ort_lat,
            "onnx_path": onnx_path,
        }

    with open(os.path.join(MET_DIR, "export_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nExport and benchmarking complete!")


if __name__ == "__main__":
    main()
