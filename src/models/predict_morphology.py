"""Standalone morphology prediction for cropped particle ROIs (Pipeline 1B).

This is Rohan's Day-7 deliverable that Aditya's ``full_pipeline.py`` imports.
The function signature is the *agreed integration contract* — do not change it
without team agreement (IPD Project Guide, Section 6, Day 7):

    predict_morphology(image_crop) -> {"morphology": str, "confidence": float}

    morphology  : one of ["sphere", "fragment", "fiber", "film", "foam"]
    confidence  : softmax probability in [0, 1]

Input ``image_crop`` is accepted as a PIL Image or a numpy/HWC RGB array, matching
Naman's ``detect_and_crop`` output. Weights are loaded automatically from
``weights/mobilenetv3_morphology_best.pth``.

NOTE: Phase 0 skeleton. The model loader is wired to ``src/models/vision_tower.py``
in Phase 3 once the trained checkpoint exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "mobilenetv3_morphology_best.pth"

MORPHOLOGY_CLASSES: tuple[str, ...] = ("sphere", "fragment", "fiber", "film", "foam")


def predict_morphology(
    image_crop: Any, weights_path: Path = DEFAULT_WEIGHTS
) -> dict[str, Any]:
    """Predict the morphology class of a single cropped particle ROI.

    Args:
        image_crop: PIL Image or numpy array (H, W, 3) in RGB — the ROI crop
            produced by ``src.detection.inference.detect_and_crop``.
        weights_path: Path to ``mobilenetv3_morphology_best.pth``.

    Returns:
        {"morphology": str, "confidence": float}

    Raises:
        NotImplementedError: until the trained checkpoint exists (Phase 3).
    """
    del image_crop, weights_path  # unused until Phase 3
    raise NotImplementedError(
        "predict_morphology() is implemented in Phase 3 (Rohan's lane) once "
        "weights/mobilenetv3_morphology_best.pth and src/models/vision_tower.py "
        "exist. See AGENT.md Section 4."
    )


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
    result = predict_morphology(args.input, args.weights)
    print(f"morphology: {result['morphology']}  confidence: {result['confidence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
