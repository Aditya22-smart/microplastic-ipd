"""Training loop for the polymer spectral classifier (Pipeline 2B).

Implements Kunsh's plan from ``IPD_Project_Guide.md`` (Section 5 — Member 2):

    Optimizer : torch.optim.NAdam(model.parameters(), lr=1e-3)
    Scheduler : CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    Loss      : nn.CrossEntropyLoss with class weights for imbalance
    Early stop: patience=10 on val macro-F1

An SVM baseline (sklearn.svm.SVC) is trained on the same preprocessed spectra;
the paired 1D-CNN vs SVM results on the FLOPP-e OOD test set feed Hypothesis H3
(Wilcoxon signed-rank across 5 seeds).

Checkpoint: ``weights/spectral_1dcnn_best.pth`` selected on val macro-F1.
Preprocessed arrays: ``data/processed/spectra/train_spectra.npy`` etc. produced
by ``src/preprocessing/spectral_preprocess.py``.

NOTE: Phase 0 contract skeleton. ``src/models/spectral_tower.py`` and the
DataLoader wiring are implemented in Kunsh's lane; the loop raises
``NotImplementedError`` until then.
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
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "spectra"
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

POLYMER_CLASSES: tuple[str, ...] = ("PE", "PP", "PS", "PMMA", "PAN")


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
    """Construct the 1D-CNN spectral classifier.

    Implemented in Kunsh's lane::

        Conv1D(1->64, k=7) -> BN -> ReLU -> MaxPool(2)
        Conv1D(64->128, k=5) -> BN -> ReLU -> MaxPool(2)
        Conv1D(128->256, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
        Linear(256->256) -> LayerNorm -> ReLU -> Linear(256->5)
    """
    del config  # unused until implementation
    raise NotImplementedError(
        "build_model() is implemented in Kunsh's lane once "
        "src/models/spectral_tower.py exists. See AGENT.md Section 4."
    )


def train_loop(model: torch.nn.Module, config: dict) -> None:
    """Run the 1D-CNN spectral training loop with NAdam + cosine schedule.

    Logs train loss, val loss, val macro-F1 and val accuracy per epoch to
    wandb; saves ``weights/spectral_1dcnn_best.pth`` on best val macro-F1.
    """
    del model, config  # unused until implementation
    raise NotImplementedError("train_loop() is implemented in Kunsh's lane.")


def train_svm_baseline(config: dict) -> None:
    """Train ``sklearn.svm.SVC`` on the same preprocessed spectra for H3."""
    del config  # unused until implementation
    raise NotImplementedError("train_svm_baseline() is implemented in Kunsh's lane.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_spectral",
        description="Train the 1D-CNN polymer classifier (Pipeline 2B).",
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
        help="Path to data/processed/spectra with *_spectra.npy / *_labels.npy.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override total epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)."
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument(
        "--svm-baseline",
        action="store_true",
        help="Also train the SVM baseline needed for Hypothesis H3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)

    print(f"config : {args.config}")
    print(f"data   : {args.data}")
    print(f"seed   : {args.seed}   wandb: {args.wandb}")
    print(f"classes: {POLYMER_CLASSES}")

    model = build_model(config)  # raises NotImplementedError until implemented
    train_loop(model, config)
    if args.svm_baseline:
        train_svm_baseline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
