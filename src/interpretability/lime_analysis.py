"""
LIME Analysis for ECG Models
==============================
Local Interpretable Model-agnostic Explanations for ECG segments.
Perturbs temporal segments of the ECG signal to understand
which regions are most important for classification.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠ lime not installed. Install with: pip install lime")


class ECGModelPredictor:
    """Wrapper for LIME compatibility — predicts probabilities."""

    def __init__(self, model, device="cpu", signal_shape=(1, 360)):
        self.model = model
        self.device = device
        self.signal_shape = signal_shape
        self.model.eval()

    def predict_proba(self, X):
        """
        X: numpy array (N, 360) - flattened ECG signals
        Returns: (N, num_classes) probabilities
        """
        X_tensor = torch.FloatTensor(X.reshape(-1, *self.signal_shape)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def run_lime_analysis(model, X_test, y_test, X_train, class_names,
                       figures_dir, model_name="model",
                       precision_tag="fp32", n_samples=3,
                       num_features=20, num_perturbations=500):
    """
    Run LIME on sample beats and save visualisations.
    """
    if not LIME_AVAILABLE:
        print("  ⚠ Skipping LIME analysis — lime not installed")
        return

    os.makedirs(figures_dir, exist_ok=True)
    device = next(model.parameters()).device

    predictor = ECGModelPredictor(model, str(device))

    # Flatten training data for LIME
    X_train_flat = X_train[:1000].reshape(-1, X_train.shape[-1])
    feature_names = [f"s_{i}" for i in range(X_train.shape[-1])]

    explainer = lime_tabular.LimeTabularExplainer(
        X_train_flat,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    for cls in range(len(class_names)):
        idx = np.where(y_test == cls)[0]
        if len(idx) == 0:
            continue

        for i, s_idx in enumerate(idx[:n_samples]):
            sample = X_test[s_idx].flatten()
            pred = predictor.predict_proba(sample[np.newaxis, :])[0].argmax()

            exp = explainer.explain_instance(
                sample,
                predictor.predict_proba,
                num_features=num_features,
                num_samples=num_perturbations,
                top_labels=1,
            )

            # Extract feature importances
            importance = np.zeros(len(sample))
            label = exp.available_labels()[0]
            for feat_idx, weight in exp.as_list(label=label):
                # Parse feature index from LIME's string format
                try:
                    parts = feat_idx.split(" ")
                    for p in parts:
                        if p.startswith("s_"):
                            idx_val = int(p.replace("s_", ""))
                            importance[idx_val] = weight
                            break
                except (ValueError, IndexError):
                    pass

            # Plot
            signal = X_test[s_idx, 0]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

            t = np.arange(len(signal))
            ax1.plot(t, signal, color="black", lw=0.8)
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"LIME — True: {class_names[cls]}, "
                          f"Pred: {class_names[pred]} "
                          f"[{model_name} {precision_tag}]")

            colors = ["green" if v > 0 else "red" for v in importance]
            ax2.bar(t, importance, color=colors, alpha=0.7, width=1.0)
            ax2.set_xlabel("Sample")
            ax2.set_ylabel("LIME Weight")
            ax2.axhline(0, color="black", lw=0.5)

            plt.tight_layout()
            save_path = os.path.join(
                figures_dir,
                f"lime_{model_name}_{precision_tag}_cls{cls}_s{i}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

    print(f"  ✅ LIME figures saved for {model_name}_{precision_tag}")
