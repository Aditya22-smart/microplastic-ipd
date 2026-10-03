"""Merge YOLO detection datasets and build a working dataset YAML.

Merges Roboflow-format datasets (train/val/test images + labels) into
data/processed/images/ and writes data/microplastic.yaml with absolute
paths, as the training scripts expect.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "microplastic.yaml"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed" / "images"

DEFAULT_CLASSES = ("microplastic", "foam", "bead")


def build_absolute_dataset_yaml(
    dataset_root: Path,
    out_path: Path,
    class_names=DEFAULT_CLASSES,
    train_split: str = "train",
    val_split: str = "valid",
    test_split: str = "test",
) -> Path:
    """Write a YOLO dataset YAML with absolute split paths."""
    dataset_root = Path(dataset_root).resolve()
    payload = {
        "path": str(dataset_root),
        "train": f"{train_split}/images",
        "val": f"{val_split}/images",
        "test": f"{test_split}/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def count_images(root: Path) -> int:
    root = Path(root)
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def merge_yolo_datasets(sources, dest=DEFAULT_PROCESSED) -> Path:
    """Merge several YOLO datasets into one train/val/test tree."""
    dest = Path(dest)
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            (dest / split / kind).mkdir(parents=True, exist_ok=True)
    for source in map(Path, sources):
        for split in ("train", "valid", "test"):
            for kind in ("images", "labels"):
                src_dir = source / split / kind
                if not src_dir.is_dir():
                    continue
                for path in src_dir.iterdir():
                    target = dest / split / kind / path.name
                    if target.exists():
                        target = dest / split / kind / f"{source.name}_{path.name}"
                    shutil.copy2(path, target)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(prog="merge_datasets")
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--dest", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    args = parser.parse_args()
    dest = merge_yolo_datasets(args.sources, dest=args.dest)
    yaml_path = build_absolute_dataset_yaml(
        dest, args.out_yaml, class_names=args.classes
    )
    print(f"merged into {dest}")
    print(f"yaml written to {yaml_path}")
    for split in ("train", "valid", "test"):
        print(f"{split}: {count_images(dest / split / 'images')} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
