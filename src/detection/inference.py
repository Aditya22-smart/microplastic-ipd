"""Detection module: detect particles and crop ROIs (Stage 1A).

Exposes detect_and_crop(image_path) -> list of dicts:
    {bbox, confidence, class, crop_img}

crop_img is a numpy HWC RGB uint8 array (the input format expected by
src.models.predict_morphology). JSON-safe variants are provided by
detect_and_crop_to_json.
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {
    "yolov8": REPO_ROOT / "weights" / "yolov8_microplastic.pt",
    "yolo26": REPO_ROOT / "weights" / "yolo26_microplastic.pt",
}
# YOLOv11 was dropped from the comparison; YOLO26 is the primary detector.
# Keep in sync with MODEL_FILES in src.detection.compare_yolo.
DEFAULT_MODEL = "yolo26"
CROP_OUTPUT_DIR = REPO_ROOT / "results" / "detection" / "crops_tmp"
#: Confidence floor applied to every detection. Fixed rather than exposed as a
#: dashboard widget so the UI, the PDF report and the JSON export all report the
#: same operating point. Chosen empirically against the score distribution of the
#: committed checkpoints -- see the note in tests/test_inference.py.
CONF_THRESHOLD = 0.5

LEGACY_CLASS_MAP = {
    "beams": "bead",
    "foams": "foam",
    "microplastic": "microplastic",
}
CANONICAL_CLASSES = ("microplastic", "foam", "bead")

_model_cache: dict[str, object] = {}


def _load_model(model_name: str = DEFAULT_MODEL):
    if model_name in _model_cache:
        return _model_cache[model_name]
    from ultralytics import YOLO

    weights_path = WEIGHTS.get(model_name)
    if weights_path is None or not weights_path.is_file():
        available = [name for name, path in WEIGHTS.items() if path.is_file()]
        if not available:
            raise FileNotFoundError("no detection weights found under weights/")
        fallback = available[0]
        warnings.warn(
            f"model {model_name!r} has no weights at {weights_path}; "
            f"falling back to {fallback!r}. Results are NOT from {model_name!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        model_name = fallback
        weights_path = WEIGHTS[model_name]
    model = YOLO(str(weights_path))
    _model_cache[model_name] = model
    return model


def _canonical_class(name: str) -> str:
    return LEGACY_CLASS_MAP.get(str(name).strip().lower(), str(name).strip().lower())


def detect_and_crop(
    image_path, model_name: str = DEFAULT_MODEL, conf_threshold: float = CONF_THRESHOLD
) -> list[dict]:
    model = _load_model(model_name)
    image = Image.open(image_path).convert("RGB")
    results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)
    output: list[dict] = []
    if not results:
        return output
    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return output
    crop_dir = CROP_OUTPUT_DIR
    crop_dir.mkdir(parents=True, exist_ok=True)
    for box in boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.width), min(y2, image.height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = np.array(image.crop((x1, y1, x2, y2)))
        crop_path = crop_dir / f"{uuid.uuid4().hex}.png"
        Image.fromarray(crop).save(crop_path)
        output.append(
            {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": round(conf, 4),
                "class": _canonical_class(
                    names[cls_idx] if isinstance(names, dict) else names[cls_idx]
                ),
                "crop_img": crop,
                "crop_path": str(crop_path),
            }
        )
    return output


def detect_and_crop_to_json(image_path, model_name: str = DEFAULT_MODEL) -> list[dict]:
    return [
        {
            "particle_id": index + 1,
            "bbox": item["bbox"],
            "detection_confidence": item["confidence"],
            "class": item["class"],
        }
        for index, item in enumerate(detect_and_crop(image_path, model_name=model_name))
    ]
