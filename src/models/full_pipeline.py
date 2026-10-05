"""Assemble the detection -> morphology and spectral pipelines (Stage 3).

Both branches are independent; they meet only at the dashboard. The visual
branch is :func:`analyze_image`, the spectral branch is :func:`analyze_spectrum`,
and the spectral branch is gated by ``MP_SPECTRAL_ENABLED``.

Every function here returns the **same 8-key record shape** regardless of branch,
so the dashboard and the PDF generator can treat a record without caring which
pipeline produced it. Keys that do not apply to a branch are ``None`` rather
than absent — an image-only record carries ``polymer: None``, which is an
honest "not computed", not a missing field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import require_spectral
from src.detection.inference import CONF_THRESHOLD, DEFAULT_MODEL, detect_and_crop
from src.models.predict_morphology import predict_morphology
from src.models.predict_polymer import predict_polymer

#: The frozen record schema. Both branches emit exactly these keys.
RECORD_KEYS: tuple[str, ...] = (
    "particle_id",
    "bbox",
    "detection_confidence",
    "class",
    "morphology",
    "morphology_confidence",
    "polymer",
    "polymer_confidence",
)


def analyze_image(
    image_path,
    *,
    conf_threshold: float | None = None,
    model_name: str | None = None,
    weights_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Detect particles in an image and classify each crop's morphology.

    This is Stage 1A (detection) followed by Stage 1B (morphology), producing
    one record per particle.

    Args:
        image_path: path to a microscope image.
        conf_threshold: YOLO confidence floor, default
            :data:`src.detection.inference.CONF_THRESHOLD`.
        model_name: which detection checkpoint to use, default
            :data:`src.detection.inference.DEFAULT_MODEL`.
        weights_path: override for the morphology checkpoint, default
            :data:`src.models.predict_morphology.DEFAULT_WEIGHTS`.

    Returns:
        A list of records with the 8 keys in :data:`RECORD_KEYS`. ``polymer``
        and ``polymer_confidence`` are always ``None`` here.
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD
    detections = detect_and_crop(
        image_path,
        model_name=DEFAULT_MODEL if model_name is None else model_name,
        conf_threshold=float(conf_threshold),
    )
    records: list[dict[str, Any]] = []
    for index, detection in enumerate(detections, start=1):
        morphology = predict_morphology(detection["crop_img"], weights_path)
        records.append(
            {
                "particle_id": index,
                "bbox": detection["bbox"],
                "detection_confidence": detection["confidence"],
                "class": detection["class"],
                "morphology": morphology["morphology"],
                "morphology_confidence": morphology["confidence"],
                "polymer": None,
                "polymer_confidence": None,
            }
        )
    return records


def analyze_spectrum(
    spectrum_array, weights_path: Path | None = None
) -> dict[str, Any]:
    """Classify a single preprocessed spectrum's polymer (Stage 2B).

    Args:
        spectrum_array: preprocessed 1D spectrum of shape [600].
        weights_path: override for the spectral checkpoint.

    Raises:
        FeatureDisabledError: if the spectral branch is switched off via
            ``MP_SPECTRAL_ENABLED``.
    """
    require_spectral("Spectral polymer classification")
    result = predict_polymer(spectrum_array, weights_path)
    return {
        "particle_id": 1,
        "bbox": None,
        "detection_confidence": None,
        "class": None,
        "morphology": None,
        "morphology_confidence": None,
        "polymer": result["polymer"],
        "polymer_confidence": result["confidence"],
    }


def run_full_pipeline(
    image_path,
    *,
    conf_threshold: float | None = None,
    model_name: str | None = None,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Run the visual branch end to end and wrap it for JSON serialisation."""
    return {
        "image": str(image_path),
        "particles": analyze_image(
            image_path,
            conf_threshold=conf_threshold,
            model_name=model_name,
            weights_path=weights_path,
        ),
    }
