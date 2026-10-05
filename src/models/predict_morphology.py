"""Morphology prediction for cropped particle ROIs (Pipeline 1B).

    predict_morphology(image_crop) -> {"morphology": str, "confidence": float}

Input ``image_crop``: PIL Image or numpy HWC RGB array.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # allow `python src/models/...py` from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from src.models.vision_tower import VisionTower  # noqa: E402
from src.preprocessing.image_preprocess import get_eval_transform  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "mobilenetv3_morphology_best.pth"

MORPHOLOGY_CLASSES: tuple[str, ...] = ("sphere", "fragment", "fiber", "film", "foam")

# Simple per-path cache so full_pipeline can classify many crops per session.
_model_cache: dict[str, tuple[VisionTower, tuple[str, ...], torch.device]] = {}


def _load_model(
    weights_path: Path,
) -> tuple[VisionTower, tuple[str, ...], torch.device]:
    """Load (once) the VisionTower from ``weights_path`` onto the best device."""
    key = str(weights_path)
    if key in _model_cache:
        return _model_cache[key]
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"weights not found at {weights_path}. Train first via "
            "`python src/training/train_morphology.py` (or the Colab notebook) "
            "and place the checkpoint before predicting."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    saved_classes = tuple(checkpoint.get("classes") or MORPHOLOGY_CLASSES)
    model = VisionTower(num_classes=len(saved_classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    _model_cache[key] = (model, saved_classes, device)
    return _model_cache[key]


def _to_rgb_uint8(image_crop: Any) -> np.ndarray:
    """Normalise a PIL Image or numpy array to a uint8 HWC RGB array."""
    if isinstance(image_crop, np.ndarray):
        array = image_crop
    else:
        convert = getattr(image_crop, "convert", None)
        if convert is None:
            raise TypeError("image_crop must be a numpy array or a PIL Image")
        array = np.array(convert("RGB"))
    if array.ndim == 2:  # grayscale -> repeat across 3 channels
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def load_morphology_model(
    weights_path: Path | None = None,
) -> tuple[VisionTower, tuple[str, ...], torch.device]:
    """Public accessor for the cached morphology model.

    Returns ``(model, classes, device)``. Used by the Grad-CAM visualiser so it
    does not need to reach into this module's private loader.
    """
    resolved = DEFAULT_WEIGHTS if weights_path is None else Path(weights_path)
    return _load_model(resolved)


def predict_morphology(
    image_crop: Any, weights_path: Path | None = None
) -> dict[str, Any]:
    """Predict the morphology class of a single cropped particle ROI.

    Args:
        image_crop: PIL Image or numpy array (H, W, 3) in RGB — the ROI crop
            produced by ``src.detection.inference.detect_and_crop``.
        weights_path: Path to ``mobilenetv3_morphology_best.pth``. Defaults to
            :data:`DEFAULT_WEIGHTS`, resolved at call time so tests can point
            the module at a temporary checkpoint.

    Returns:
        {"morphology": str, "confidence": float}
    """
    # Resolve inside the body, not as a default argument: a default is bound at
    # import time, so patching DEFAULT_WEIGHTS in a test would silently do
    # nothing and the real checkpoint would load instead.
    resolved = DEFAULT_WEIGHTS if weights_path is None else Path(weights_path)
    model, classes, device = _load_model(resolved)
    transform = get_eval_transform(224)
    tensor = transform(image=_to_rgb_uint8(image_crop))["image"]

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probabilities))
    return {
        "morphology": classes[index],
        "confidence": float(probabilities[index].cpu()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_morphology",
        description="Classify the morphology of a single particle ROI image.",
    )
    parser.add_argument("--input", required=True, help="Path to the crop image.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to mobilenetv3_morphology_best.pth.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from PIL import Image

    with Image.open(args.input) as image_handle:
        image = image_handle.convert("RGB")
    result = predict_morphology(image, args.weights)
    print(f"morphology: {result['morphology']}  confidence: {result['confidence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
