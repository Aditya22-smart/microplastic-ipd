"""Compare YOLOv8 / YOLOv11 / YOLO26 on the merged test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "microplastic.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "detection" / "yolo_comparison_table.csv"

MODEL_FILES = {
    "yolov8s": REPO_ROOT / "weights" / "yolov8s_microplastic.pt",
    "yolo11s": REPO_ROOT / "weights" / "yolo11s_microplastic.pt",
    "yolo26s": REPO_ROOT / "weights" / "yolo26s_microplastic.pt",
}


def evaluate_model(name: str, weights: Path, data: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data))
    return {
        "model": name,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "params": int(sum(p.numel() for p in model.model.parameters())),
        "inference_ms": round(float(metrics.speed["inference"]), 2),
    }


def build_table(data: Path = DEFAULT_DATA, out: Path = DEFAULT_OUT, models: dict | None = None) -> list[dict]:
    rows = []
    for name, weights in (models or MODEL_FILES).items():
        if not Path(weights).is_file():
            print(f"skipping {name}: {weights} not found")
            continue
        rows.append(evaluate_model(name, Path(weights), Path(data)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare_yolo")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_table(data=args.data, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
