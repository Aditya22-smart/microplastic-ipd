"""Standalone polymer prediction for a preprocessed spectrum (Pipeline 2B).

This is Kunsh's Day-9 deliverable that Aditya's ``full_pipeline.py`` imports.
The function signature is the *agreed integration contract* — do not change it
without team agreement (IPD Project Guide, Section 5, Day 9):

    predict_polymer(spectrum_array) -> {"polymer": str, "confidence": float}

    polymer    : one of ["PE", "PP", "PS", "PMMA", "PAN"]
    confidence : softmax probability in [0, 1]

Input ``spectrum_array`` is the *preprocessed* 1D spectrum ([600] float array —
Savitzky-Golay -> water-vapour zeroing -> SNV, ready for the model). Weights and
the fitted scaler are loaded automatically from the ``weights/`` folder.

NOTE: Phase 0 skeleton. The model loader is wired to ``src/models/spectral_tower.py``
in Kunsh's lane once the trained checkpoint exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "spectral_1dcnn_best.pth"

POLYMER_CLASSES: tuple[str, ...] = ("PE", "PP", "PS", "PMMA", "PAN")


def predict_polymer(
    spectrum_array: Any, weights_path: Path = DEFAULT_WEIGHTS
) -> dict[str, Any]:
    """Predict the polymer type of a single preprocessed spectrum.

    Args:
        spectrum_array: 1D numpy array of shape [600] — already preprocessed
            by ``src.preprocessing.spectral_preprocess``.
        weights_path: Path to ``spectral_1dcnn_best.pth``.

    Returns:
        {"polymer": str, "confidence": float}

    Raises:
        NotImplementedError: until the trained checkpoint exists (Kunsh's lane).
    """
    del spectrum_array, weights_path  # unused until implementation
    raise NotImplementedError(
        "predict_polymer() is implemented in Kunsh's lane once "
        "weights/spectral_1dcnn_best.pth and src/models/spectral_tower.py exist. "
        "See AGENT.md Section 4."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_polymer",
        description="Predict the polymer type of a preprocessed spectrum.",
    )
    parser.add_argument(
        "--input", required=True, help="Path to a sample spectrum CSV/txt."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to spectral_1dcnn_best.pth.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = predict_polymer(args.input, args.weights)
    print(f"polymer: {result['polymer']}  confidence: {result['confidence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
