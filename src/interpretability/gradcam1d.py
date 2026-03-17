"""
Grad-CAM for 1D Convolutional Networks
========================================
Adapted Grad-CAM that highlights which temporal regions of the ECG
signal most influence the model's prediction.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


class GradCAM1D:
    """
    Grad-CAM for 1D convolutional networks.

    Computes class-discriminative localisation map by combining
    feature maps from a target conv layer with gradient-based
    importance weights.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model (must have Conv1D layers)
            target_layer: the nn.Conv1d layer to use for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            x: input tensor (1, 1, 360)
            target_class: class index (None = use predicted class)

        Returns:
            cam: numpy array (360,) — normalised heatmap
            pred_class: predicted class index
        """
        self.model.eval()
        x.requires_grad_(True)

        output = self.model(x)
        pred_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average of gradients → channel weights
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)

        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1)  # (1, L)
        cam = F.relu(cam)  # Only positive contributions

        # Upsample to input length
        cam = F.interpolate(cam.unsqueeze(0), size=x.shape[2],
                            mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class


def plot_gradcam(signal, cam, true_class, pred_class, class_names,
                 save_path=None, title_extra=""):
    """Plot ECG signal with Grad-CAM overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    t = np.arange(len(signal))

    # ECG signal
    ax1.plot(t, signal, color="black", lw=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"ECG Beat — True: {class_names[true_class]}, "
                  f"Pred: {class_names[pred_class]} {title_extra}")

    # Grad-CAM heatmap overlay
    ax2.plot(t, signal, color="black", lw=0.5, alpha=0.5)
    ax2.fill_between(t, signal.min(), signal.max(), where=cam > 0.3,
                     alpha=0.3, color="red", label="Grad-CAM")
    im = ax2.imshow(cam[np.newaxis, :], aspect="auto", cmap="jet",
                    extent=[0, len(signal), signal.min(), signal.max()],
                    alpha=0.6)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Amplitude")
    plt.colorbar(im, ax=ax2, label="Grad-CAM Intensity")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_gradcam_analysis(model, X_test, y_test, target_layer,
                          class_names, figures_dir, model_name="model",
                          precision_tag="fp32", n_samples=3):
    """
    Run Grad-CAM on sample beats from each class.

    Args:
        model: trained PyTorch model
        X_test: test data (N, 1, 360)
        y_test: test labels
        target_layer: Conv1D layer for Grad-CAM
        class_names: list of class name strings
        figures_dir: directory to save figures
        model_name: model identifier
        precision_tag: precision level tag
        n_samples: samples per class
    """
    os.makedirs(figures_dir, exist_ok=True)
    device = next(model.parameters()).device
    grad_cam = GradCAM1D(model, target_layer)

    num_classes = len(class_names)

    for cls in range(num_classes):
        idx = np.where(y_test == cls)[0]
        if len(idx) == 0:
            continue

        samples = idx[:n_samples]
        for i, s_idx in enumerate(samples):
            x = torch.FloatTensor(X_test[s_idx:s_idx+1]).to(device)
            cam, pred = grad_cam.generate(x)
            signal = X_test[s_idx, 0]

            save_path = os.path.join(
                figures_dir,
                f"gradcam_{model_name}_{precision_tag}_cls{cls}_s{i}.png"
            )
            plot_gradcam(signal, cam, cls, pred, class_names,
                         save_path=save_path,
                         title_extra=f"[{model_name} {precision_tag}]")

    print(f"  ✅ Grad-CAM figures saved for {model_name}_{precision_tag}")
