"""Assemble detection -> morphology and spectral pipelines (Stage 3)."""

from __future__ import annotations

from typing import Any

from src.detection.inference import detect_and_crop
from src.models.predict_morphology import predict_morphology
from src.models.predict_polymer import predict_polymer


def analyze_image(image_path) -> list[dict[str, Any]]:
    """Detect particles in an image and classify each crop's morphology."""
    records = []
    for index, detection in enumerate(detect_and_crop(image_path), start=1):
        morphology = predict_morphology(detection["crop_img"])
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


def analyze_spectrum(spectrum_array) -> dict[str, Any]:
    """Classify a single preprocessed spectrum's polymer."""
    result = predict_polymer(spectrum_array)
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


def run_full_pipeline(image_path) -> dict[str, Any]:
    return {"image": str(image_path), "particles": analyze_image(image_path)}
