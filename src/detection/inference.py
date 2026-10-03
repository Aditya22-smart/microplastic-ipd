"""
Detection module for the microplastic-ipd pipeline.

Exposes detect_and_crop(image_path) -> list of dicts:
    {bbox, confidence, class, crop_img}

This is the critical dependency Rohan's morphology classifier
and Aditya's full_pipeline.py build on top of.
"""

import os
import uuid
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[2]  # src/detection/ -> repo root
WEIGHTS = {
    "yolov26": REPO_ROOT / "weights" / "yolov26_microplastic.pt",
    "yolov8": REPO_ROOT / "weights" / "yolov8_microplastic.pt",
}
DEFAULT_MODEL = "yolov26"  # best performing model; change to "yolov8" if needed
CROP_OUTPUT_DIR = REPO_ROOT / "results" / "detection" / "crops_tmp"
CONF_THRESHOLD = 0.25

CLASS_NAMES = ["Beams", "Foams", "Microplastic"]

# ---- Lazy-loaded models (cached per model_name so both can be used without reloading) ----
_models = {}


def _get_model(model_name=DEFAULT_MODEL):
    if model_name not in WEIGHTS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(WEIGHTS)}.")

    if model_name not in _models:
        weights_path = WEIGHTS[model_name]
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                "Make sure the trained .pt file is committed to weights/."
            )
        _models[model_name] = YOLO(str(weights_path))

    return _models[model_name]


def detect_and_crop(image_path, model_name=DEFAULT_MODEL, conf=CONF_THRESHOLD, save_crops=True):
    """
    Run detection on a single image and return cropped particle ROIs.

    Args:
        image_path (str): path to the input microscope image
        model_name (str): "yolov26" (default) or "yolov8"
        conf (float): confidence threshold for detections
        save_crops (bool): whether to save crop images to disk

    Returns:
        List[dict]: one entry per detected particle, each with:
            - bbox: [x1, y1, x2, y2] (pixel coords, ints)
            - confidence: float
            - class: str (class name)
            - class_id: int
            - crop_img: PIL.Image.Image (the cropped particle)
            - crop_path: str or None (saved path, if save_crops=True)
    """
    model = _get_model(model_name)
    image_path = str(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(source=image_path, conf=conf, verbose=False)
    r = results[0]  # single image -> single result

    original_img = Image.open(image_path).convert("RGB")
    detections = []

    if save_crops:
        CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)

        crop_img = original_img.crop((x1, y1, x2, y2))

        crop_path = None
        if save_crops:
            crop_filename = f"{uuid.uuid4().hex[:8]}_{class_name}.jpg"
            crop_path = str(CROP_OUTPUT_DIR / crop_filename)
            crop_img.save(crop_path)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": round(confidence, 4),
            "class": class_name,
            "class_id": class_id,
            "crop_img": crop_img,
            "crop_path": crop_path,
        })

    return detections


def detect_and_crop_to_json(image_path, model_name=DEFAULT_MODEL, conf=CONF_THRESHOLD):
    """
    Same as detect_and_crop, but returns a JSON-serialisable version
    (drops the PIL Image object, keeps only the saved path).
    Used by the dashboard / full_pipeline.py integration.
    """
    detections = detect_and_crop(image_path, model_name=model_name, conf=conf, save_crops=True)
    return [
        {
            "bbox": d["bbox"],
            "confidence": d["confidence"],
            "class": d["class"],
            "class_id": d["class_id"],
            "crop_path": d["crop_path"],
        }
        for d in detections
    ]


if __name__ == "__main__":
    # Quick manual smoke test:
    # python src/detection/inference.py path/to/image.jpg [yolov26|yolov8]
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_name]")
        sys.exit(1)

    img_path = sys.argv[1]
    model_choice = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    out = detect_and_crop(img_path, model_name=model_choice)
    print(f"Model: {model_choice} | Found {len(out)} particles:")
    for d in out:
        print(f"  {d['class']} ({d['confidence']:.2f}) @ {d['bbox']} -> {d['crop_path']}")