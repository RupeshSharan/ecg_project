"""
Model accuracy report utility.

Reads all *_metrics.json files from results/metrics, prints a sorted table,
and optionally exports a CSV summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_metric_rows(metrics_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping unreadable file: {path.name} ({exc})")
            continue

        row = {
            "file": path.name,
            "model": payload.get("model", "unknown"),
            "precision": payload.get("precision", "unknown"),
            "accuracy": _safe_float(payload.get("accuracy", 0.0)),
            "f1_macro": _safe_float(payload.get("f1_macro", 0.0)),
            "auc_roc": _safe_float(payload.get("auc_roc", 0.0)),
            "latency_ms": _safe_float(payload.get("latency_ms", {}).get("mean_ms", 0.0)),
            "model_size_mb": _safe_float(payload.get("model_size_mb", 0.0)),
            "total_params": int(payload.get("total_params", 0) or 0),
        }
        rows.append(row)
    return rows


def print_table(rows: List[Dict], top_n: int | None = None) -> None:
    if not rows:
        print("No metrics rows found.")
        return

    rows = sorted(rows, key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    if top_n is not None and top_n > 0:
        rows = rows[:top_n]

    header = (
        f"{'MODEL':<14} {'PRECISION':<16} {'ACC':>8} {'F1':>8} {'AUC':>8} "
        f"{'LAT(ms)':>10} {'SIZE(MB)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<14} {r['precision']:<16} "
            f"{r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {r['auc_roc']:>8.4f} "
            f"{r['latency_ms']:>10.2f} {r['model_size_mb']:>10.2f}"
        )


def write_csv(rows: List[Dict], output_path: Path) -> None:
    import csv

    fieldnames = [
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (-r["accuracy"], -r["f1_macro"])):
            writer.writerow(row)
    print(f"Saved CSV summary: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model accuracy summary from metrics JSON files.")
    parser.add_argument(
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument("--top", type=int, default=0, help="Show top N rows by accuracy (0 = all).")
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional output CSV path (e.g. results/metrics/accuracy_summary.csv).",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    rows = load_metric_rows(metrics_dir)
    if not rows:
        print("No *_metrics.json files found.")
        return

    print(f"Metrics source: {metrics_dir}")
    print(f"Total rows: {len(rows)}")
    print_table(rows, top_n=(args.top if args.top > 0 else None))

    if args.save_csv:
        write_csv(rows, Path(args.save_csv).resolve())


if __name__ == "__main__":
    main()
