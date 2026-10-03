"""Train YOLOv11 on the merged microplastic detection dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "yolo11.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train(config_path: Path = DEFAULT_CONFIG, *, copy_to_local: bool = False):
    from ultralytics import YOLO

    config = load_config(config_path)
    model_name = config.get("model", "yolo11s.pt")
    data = str(config["data"])
    if copy_to_local:
        import shutil
        import tempfile

        yaml_path = Path(data)
        dataset_root = Path(
            yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get(
                "path", yaml_path.parent
            )
        )
        if not dataset_root.is_absolute():
            dataset_root = (yaml_path.parent / dataset_root).resolve()
        tmp = Path(tempfile.mkdtemp(prefix="yolo11_data_"))
        shutil.copytree(dataset_root, tmp / dataset_root.name, dirs_exist_ok=True)
        new_yaml = tmp / "microplastic.yaml"
        new_yaml.write_text(
            yaml_path.read_text(encoding="utf-8").replace(
                str(dataset_root), str(tmp / dataset_root.name)
            ),
            encoding="utf-8",
        )
        data = str(new_yaml)
    model = YOLO(model_name)
    results = model.train(
        data=data,
        epochs=int(config.get("epochs", 100)),
        batch=int(config.get("batch", 16)),
        imgsz=int(config.get("imgsz", 640)),
        optimizer=config.get("optimizer", "AdamW"),
        lr0=float(config.get("lr0", 0.01)),
        seed=int(config.get("seed", 42)),
        project=str(config.get("project", "results/detection")),
        name=config.get("name", "yolo11s_microplastic"),
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(prog="train_yolo11")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--copy-to-local", action="store_true")
    args = parser.parse_args()
    train(args.config, copy_to_local=args.copy_to_local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
