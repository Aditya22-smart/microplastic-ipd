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
    # `turbo`, not `jet`. `jet` is matplotlib's historical default and is the
    # textbook example of a bad colormap: it is not perceptually uniform, so it
    # invents banding and false edges that are not in the CAM, and its green
    # midtones disappear for red-green colour blindness. `turbo` was designed as
    # a jet replacement with near-uniform luminance and a colour-vision-safe
    # path. Both run cool (blue) to hot (red), so the reading is unchanged.
    heatmap = (cm.turbo(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(image)
    blended = (alpha * heatmap + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def compute_spectral_saliency(model, spectrum, target_class=None, device=None):
    """Vanilla gradient saliency for spectral 1D-CNN (Pipeline 2B - Day 8)."""
    from src.evaluation.spectral_saliency import compute_spectral_saliency as _css

    return _css(model, spectrum, target_class=target_class, device=device)

def grad_cam_for_crop(crop, weights_path: Path | None = None) -> np.ndarray:
    """Run Grad-CAM on one cropped particle ROI.

    Wraps the model loading and input transform so callers (notably the
    dashboard) do not have to import torch plumbing. The ROI is re-cropped from
    the original image by the caller, so this costs no extra detection pass.

    Args:
        crop: a PIL Image of a single particle.
        weights_path: morphology checkpoint; defaults to
            :data:`src.models.predict_morphology.DEFAULT_WEIGHTS`, resolved at
            call time.

    Returns:
        A ``[H, W]`` CAM normalised to ``[0, 1]``.
    """
    from src.models.predict_morphology import load_morphology_model
    from src.preprocessing.image_preprocess import get_eval_transform

    model, _classes, device = load_morphology_model(weights_path)
    transform = get_eval_transform(224)
    tensor = transform(image=np.asarray(crop.convert("RGB")))["image"]
    cam = GradCAM(model, default_target_layer(model))
    try:
        return cam.generate(tensor.unsqueeze(0).to(device))
    finally:
        # Always unhook, or every widget interaction leaks another forward hook.
        cam.close()
