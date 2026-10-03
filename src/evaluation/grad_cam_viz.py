"""Grad-CAM visualisation for the morphology classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "classification" / "gradcam_morphology"


def default_target_layer(model) -> nn.Module:
    """Return the last convolutional block of ``model`` — a safe Grad-CAM target.

    Deliberately skips pooling/head layers, whose output has no spatial extent.
    """
    backbone = getattr(model, "backbone", model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError(
            f"{type(backbone).__name__} has no `.blocks`; pass an explicit "
            "convolutional layer to GradCAM instead."
        )
    return blocks[-1]


class GradCAM:
    """Minimal Grad-CAM for a single conv layer.

    Gradients are captured with a *tensor* hook installed from the forward hook,
    not with ``register_full_backward_hook``. The latter wraps the layer output
    in a ``BackwardHookFunction``, which forbids the in-place activation that
    timm >= 1.0 MobileNetV3 applies (``act2`` is an in-place Hardswish) and
    crashes with "Output 0 of BackwardHookFunction is a view and is being
    modified inplace".
    """

    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._hooks = [target_layer.register_forward_hook(self._forward_hook)]

    def _forward_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return  # nothing to hook (tuple output)
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> np.ndarray:
        """Return a ``[H, W]`` CAM in ``[0, 1]`` for a single input image.

        ``input_tensor`` must be one image, ``[1, 3, H, W]``. The target layer
        must be convolutional and *not* already pooled — on timm >= 1.0
        MobileNetV3 the global average pool runs inside ``forward_features``, so
        ``conv_head`` emits ``[B, 1024, 1, 1]``. Hook ``backbone.blocks[-1]``.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"expected a single image [1, 3, H, W], got {tuple(input_tensor.shape)}"
            )
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hooks never fired — the target layer was probably not "
                "used in this model's forward pass."
            )
        if self.activations.ndim != 4:
            raise ValueError(
                "Grad-CAM target layer did not emit a 4D feature map; pick a "
                "conv layer, not a pooling/classifier layer."
            )
        if self.activations.shape[-2] * self.activations.shape[-1] == 1:
            raise ValueError(
                "Grad-CAM target layer is already spatially pooled to 1x1, so "
                "there is nothing to visualise. Hook a conv block before the "
                "pool instead, e.g. `model.backbone.blocks[-1]`."
            )
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1))  # [1, H, W]
        cam = heatmap[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()


def overlay_cam_on_image(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    import matplotlib.cm as cm

    cam_resized = (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                image.size, Image.Resampling.BILINEAR
            )
        )
        / 255.0
    )
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(image)
    blended = (alpha * heatmap + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)
