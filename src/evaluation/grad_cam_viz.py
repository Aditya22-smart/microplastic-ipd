"""Grad-CAM visualisation for the morphology classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "classification" / "gradcam_morphology"


class GradCAM:
    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    import matplotlib.cm as cm

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)) / 255.0
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(image)
    blended = (alpha * heatmap + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)
