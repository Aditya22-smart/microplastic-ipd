"""Tests for the detection module (Stage 1A).

The parts that matter without a checkpoint are the class-name canonicalisation
and the JSON projection. The Roboflow-trained checkpoints label their classes
``Beams`` / ``Foams`` / ``Microplastic``; everything downstream reads the
canonical ``bead`` / ``foam`` / ``microplastic``, so that mapping is pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detection.inference import (
    CANONICAL_CLASSES,
    CONF_THRESHOLD,
    DEFAULT_MODEL,
    LEGACY_CLASS_MAP,
    WEIGHTS,
    _canonical_class,
    detect_and_crop_to_json,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Beams", "bead"),
        ("beams", "bead"),
        ("  BEAMS  ", "bead"),
        ("Foams", "foam"),
        ("foams", "foam"),
        ("Microplastic", "microplastic"),
        ("microplastic", "microplastic"),
    ],
)
def test_legacy_roboflow_names_are_canonicalised(raw: str, expected: str) -> None:
    assert _canonical_class(raw) == expected


def test_unknown_class_names_pass_through_lowercased() -> None:
    """An unseen class must not be silently remapped to something wrong."""
    assert _canonical_class("Unknown Debris") == "unknown debris"
    assert _canonical_class("Unknown Debris") not in CANONICAL_CLASSES


def test_every_legacy_key_maps_into_the_canonical_vocabulary() -> None:
    for legacy, canonical in LEGACY_CLASS_MAP.items():
        assert canonical in CANONICAL_CLASSES
        assert _canonical_class(legacy) == canonical


def test_legacy_map_is_case_insensitive_because_canonical_class_lowercases() -> None:
    """Guard the invariant: the map must only be consulted with lowercase keys."""
    for legacy in LEGACY_CLASS_MAP:
        assert (
            legacy == legacy.lower()
        ), f"{legacy!r} is not lowercase, so _canonical_class would never match it"


def test_default_model_is_a_real_weights_key() -> None:
    assert DEFAULT_MODEL in WEIGHTS
    assert WEIGHTS[DEFAULT_MODEL].is_absolute()


def test_every_weights_entry_points_at_a_microplastic_checkpoint() -> None:
    for name, path in WEIGHTS.items():
        assert path.name.endswith("_microplastic.pt"), f"{name} -> {path.name}"


def test_json_projection_is_serialisable_and_carries_no_pil_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dashboard and PDF consume this dict; a leaked PIL Image breaks both."""
    import json

    import src.detection.inference as inference

    monkeypatch.setattr(
        inference,
        "detect_and_crop",
        lambda image_path, model_name="yolo26": [
            {
                "bbox": [1, 2, 3, 4],
                "confidence": 0.5,
                "class": "bead",
                "crop_img": object(),  # deliberately not JSON-safe
                "crop_path": "/tmp/x.png",
            }
        ],
    )
    records = detect_and_crop_to_json("unused.png")
    assert tuple(records[0]) == (
        "particle_id",
        "bbox",
        "detection_confidence",
        "class",
    )
    json.dumps(records)  # must not raise


def test_default_confidence_threshold_is_the_documented_one() -> None:
    """Pinned at 0.50.

    This is a project-wide operating point, not a tuned hyperparameter: the
    dashboard, the PDF and the JSON export all inherit it, so changing it silently
    changes every previously reported result. Raise it only alongside a
    precision/recall measurement, not on impression.

    The committed checkpoints score low -- mean detection confidence on the test
    field is ~0.49 -- so 0.50 is at the upper end of useful. If a real image
    reports zero particles, this constant is the first thing to revisit.
    """
    assert CONF_THRESHOLD == 0.50
