"""Tests for the visual-branch integration seam.

``full_pipeline`` is where Stage 1A detection and Stage 1B morphology are joined,
and it is also the module that enforces the frozen 8-key record schema that the
dashboard and the PDF generator both depend on. These tests stub out the two
model calls so they run without checkpoints; the heavyweight end-to-end path is
covered by ``tests/test_dashboard.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import SPECTRAL_ENABLED_ENV, FeatureDisabledError
from src.models import full_pipeline as fp
from src.models.full_pipeline import RECORD_KEYS, analyze_image, analyze_spectrum


@pytest.fixture
def stub_detection(monkeypatch: pytest.MonkeyPatch):
    """Replace detection with two fixed boxes so the tests stay deterministic."""
    calls: list[dict] = []

    def fake_detect_and_crop(image_path, model_name="yolo26", conf_threshold=0.25):
        calls.append({"model_name": model_name, "conf_threshold": conf_threshold})
        return [
            {
                "bbox": [10, 20, 30, 40],
                "confidence": 0.91,
                "class": "bead",
                "crop_img": "crop-a",
            },
            {
                "bbox": [50, 60, 70, 80],
                "confidence": 0.72,
                "class": "foam",
                "crop_img": "crop-b",
            },
        ]

    monkeypatch.setattr(fp, "detect_and_crop", fake_detect_and_crop)
    return calls


@pytest.fixture
def stub_morphology(monkeypatch: pytest.MonkeyPatch):
    """Replace the morphology classifier, echoing the crop it was handed."""
    seen: list[str] = []

    def fake_predict_morphology(image_crop, weights_path=None):
        seen.append(str(weights_path))
        return {"morphology": "sphere", "confidence": 0.5 + len(seen) / 100}

    monkeypatch.setattr(fp, "predict_morphology", fake_predict_morphology)
    return seen


def test_record_schema_is_frozen_and_identical_on_both_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dashboard contract: both branches emit exactly RECORD_KEYS."""
    monkeypatch.setattr(fp, "detect_and_crop", lambda *a, **k: [])
    records = analyze_image("unused.png")
    assert all(tuple(record) == RECORD_KEYS for record in records)

    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "1")
    monkeypatch.setattr(
        fp, "predict_polymer", lambda s, w=None: {"polymer": "PE", "confidence": 0.8}
    )
    spectrum_record = analyze_spectrum([0.0] * 600)
    assert tuple(spectrum_record) == RECORD_KEYS


def test_image_records_never_claim_a_polymer(stub_detection, stub_morphology) -> None:
    for record in analyze_image("unused.png"):
        assert record["polymer"] is None
        assert record["polymer_confidence"] is None


def test_particle_ids_are_one_based_and_sequential(
    stub_detection, stub_morphology
) -> None:
    records = analyze_image("unused.png")
    assert [record["particle_id"] for record in records] == [1, 2]


def test_detection_arguments_are_forwarded(stub_detection, stub_morphology) -> None:
    analyze_image("unused.png", conf_threshold=0.4, model_name="yolov8")
    assert stub_detection == [{"model_name": "yolov8", "conf_threshold": 0.4}]


def test_defaults_are_used_when_no_overrides_are_given(
    stub_detection, stub_morphology
) -> None:
    from src.detection.inference import CONF_THRESHOLD, DEFAULT_MODEL

    analyze_image("unused.png")
    assert stub_detection[0]["model_name"] == DEFAULT_MODEL
    assert stub_detection[0]["conf_threshold"] == CONF_THRESHOLD


def test_weights_path_override_reaches_the_classifier(
    stub_detection, stub_morphology
) -> None:
    override = Path("/tmp/some-other-morphology.pth")
    analyze_image("unused.png", weights_path=override)
    assert stub_morphology == [str(override)] * 2


def test_analyze_spectrum_is_blocked_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SPECTRAL_ENABLED_ENV, raising=False)
    with pytest.raises(FeatureDisabledError, match="MP_SPECTRAL_ENABLED"):
        analyze_spectrum([0.0] * 600)


def test_analyze_spectrum_defers_to_the_model_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "1")
    monkeypatch.setattr(
        fp, "predict_polymer", lambda s, w=None: {"polymer": "PET", "confidence": 0.77}
    )
    record = analyze_spectrum([0.0] * 600)
    assert record["polymer"] == "PET"
    assert record["morphology"] is None


def test_run_full_pipeline_wraps_particles_under_a_stable_key(
    stub_detection, stub_morphology
) -> None:
    result = fp.run_full_pipeline("some/image.png")
    assert set(result) == {"image", "particles"}
    assert result["image"] == "some/image.png"
    assert len(result["particles"]) == 2
