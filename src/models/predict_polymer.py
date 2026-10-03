"""Standalone polymer prediction for a preprocessed spectrum (Pipeline 2B).

    predict_polymer(spectrum_array) -> {"polymer": str, "confidence": float}

Input is a preprocessed 1D spectrum of shape [600] (Savitzky-Golay ->
water-vapour zeroing -> SNV), ready for the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.spectral_tower import SpectralTower

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "spectral_1dcnn_best.pth"

POLYMER_CLASSES: tuple[str, ...] = ("PE", "PP", "PS", "PMMA", "PAN")

_model_cache: dict[str, tuple[SpectralTower, list[str], torch.device]] = {}


def _load_model(weights_path: Path) -> tuple[SpectralTower, list[str], torch.device]:
    key = str(weights_path)
    if key in _model_cache:
        return _model_cache[key]
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found at {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    classes = checkpoint.get("classes", list(POLYMER_CLASSES))
    model = SpectralTower(num_classes=len(classes), input_bands=checkpoint.get("input_bands", 600))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    _model_cache[key] = (model, classes, device)
    return _model_cache[key]


def predict_polymer(
    spectrum_array: Any, weights_path: Path = DEFAULT_WEIGHTS
) -> dict[str, Any]:
    model, classes, device = _load_model(weights_path)
    spectrum = np.asarray(spectrum_array, dtype=np.float32)
    if spectrum.ndim == 1:
        spectrum = spectrum[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(spectrum).to(device))
    probabilities = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probabilities))
    return {"polymer": classes[index], "confidence": float(probabilities[index].cpu())}


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
