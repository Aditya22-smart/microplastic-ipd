"""
Unit test for src/detection/inference.py
Run with: pytest tests/test_detection.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.inference import detect_and_crop, detect_and_crop_to_json

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_images"

# Point these at 3 real images from your test set
SAMPLE_IMAGES = [
    str(SAMPLE_IMAGES_DIR / "sample1.jpg"),
    str(SAMPLE_IMAGES_DIR / "sample2.jpg"),
    str(SAMPLE_IMAGES_DIR / "sample3.jpg"),
]


@pytest.mark.parametrize("image_path", SAMPLE_IMAGES)
def test_detect_and_crop_runs(image_path):
    assert os.path.exists(image_path), f"Missing sample image: {image_path}"

    detections = detect_and_crop(image_path, save_crops=True)

    assert isinstance(detections, list)

    for d in detections:
        assert "bbox" in d and len(d["bbox"]) == 4
        assert "confidence" in d and 0.0 <= d["confidence"] <= 1.0
        assert "class" in d and isinstance(d["class"], str)
        assert "crop_img" in d
        assert d["crop_img"].size[0] > 0 and d["crop_img"].size[1] > 0


@pytest.mark.parametrize("image_path", SAMPLE_IMAGES)
def test_detect_and_crop_to_json_is_serialisable(image_path):
    import json

    result = detect_and_crop_to_json(image_path)
    # Should not raise
    json_str = json.dumps(result)
    assert isinstance(json_str, str)


def test_bbox_within_image_bounds():
    from PIL import Image

    image_path = SAMPLE_IMAGES[0]
    img = Image.open(image_path)
    w, h = img.size

    detections = detect_and_crop(image_path)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h