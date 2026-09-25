"""Training loop for the morphology classifier (Pipeline 1B).

Implements Rohan's progressive-unfreezing schedule from ``IPD_Project_Guide.md``
(Section 6 — Member 3):

    Epochs   1-10 : backbone frozen — train the projection head only (lr=1e-3)
    Epochs  11-30 : unfreeze last 4 backbone blocks (backbone lr=1e-4, head 1e-3)
    Epochs  31-50 : full fine-tune of all layers (lr=1e-5)

Loss: ``torch.nn.CrossEntropyLoss`` weighted by ``class_weights.json``
(inverse-frequency weights produced by ``download_data.py organize``).

Optimizer: ``torch.optim.NAdam``, scheduler ``CosineAnnealingLR``.

Checkpoint: ``weights/mobilenetv3_morphology_best.pth`` selected on val macro-F1.

Metrics: wandb logs train/val loss, val accuracy and val macro-F1 per epoch.
A ResNet18 baseline (same settings) is trained for the H2 McNemar comparison.

NOTE: This file is the Phase 0 contract pin. The model construction
(``src/models/vision_tower.py``) and the DataLoader wiring are implemented in
Phase 3 (Rohan's lane). Until then the training loop raises ``NotImplementedError``.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training_config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "morphology"
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MORPHOLOGY_CLASSES: tuple[str, ...] = ("sphere", "fragment", "fiber", "film", "foam")


def set_seed(seed: int) -> None:
    """Seed torch/numpy/random for reproducible runs (see Guide, Section 5)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Load the shared training YAML (configs/training_config.yaml)."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> torch.nn.Module:
    """Construct the MobileNetV3-Small morphology classifier.

    Implemented in Phase 3::

        timm.create_model("mobilenetv3_small_100", pretrained=True,
                          num_classes=0, global_pool="avg")
        + projection head Linear(576->256) -> LayerNorm -> ReLU -> Linear(256->5)
    """
    del config  # unused until Phase 3
    raise NotImplementedError(
        "build_model() is implemented in Phase 3 (Rohan's lane) once "
        "src/models/vision_tower.py exists. See AGENT.md Section 4."
    )


def train_loop(model: torch.nn.Module, config: dict) -> None:
    """Run the three-stage progressive-unfreezing schedule.

    Stage A (epochs 1-10)  : frozen backbone, head lr=1e-3
    Stage B (epochs 11-30) : last 4 blocks unfrozen, block lr=1e-4, head lr=1e-3
    Stage C (epochs 31-50) : all layers, lr=1e-5

    Tracks val macro-F1 and saves ``weights/mobilenetv3_morphology_best.pth``.
    """
    del model, config  # unused until Phase 3
    raise NotImplementedError("train_loop() is implemented in Phase 3 (Rohan's lane).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_morphology",
        description="Train the MobileNetV3 morphology classifier (Pipeline 1B).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configs/training_config.yaml.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to data/processed/morphology with train/val/test class folders.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override total epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)."
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument(
        "--resnet-baseline",
        action="store_true",
        help="Also train a ResNet18 baseline for the H2 McNemar comparison.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)

    print(f"config : {args.config}")
    print(f"data   : {args.data}")
    print(f"seed   : {args.seed}   wandb: {args.wandb}")
    print(f"classes: {MORPHOLOGY_CLASSES}")

    model = build_model(config)  # raises NotImplementedError until Phase 3
    train_loop(model, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
