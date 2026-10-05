"""Training loop for the polymer spectral classifier (Pipeline 2B).

    Optimizer : torch.optim.NAdam(model.parameters(), lr=1e-3)
    Scheduler : CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    Loss      : nn.CrossEntropyLoss with class weights
    Early stop: patience=10 on val macro-F1

An SVM baseline (sklearn.svm.SVC) is trained on the same preprocessed spectra
for Hypothesis H3.

Checkpoint: weights/spectral_1dcnn_best.pth (selected on val macro-F1).
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training_config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "spectra"
WEIGHTS_DIR = REPO_ROOT / "weights"

POLYMER_CLASSES: tuple[str, ...] = ("PE", "PP", "PS", "PMMA", "PAN")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> torch.nn.Module:
    from src.models.spectral_tower import SpectralTower

    input_bands = int(config.get("model", {}).get("spectral_input_bands", 600))
    classes = config.get("data", {}).get("polymer_classes") or list(POLYMER_CLASSES)
    return SpectralTower(num_classes=len(classes), input_bands=input_bands)


def _load_arrays(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the train/validation arrays written by ``spectral_preprocess``.

    A dedicated validation split is **required**. The earlier version fell back
    to a random split over the pixels of ``train_spectra.npy``. Because adjacent
    pixels in a hyperspectral field are near-identical, a pixel-level split puts
    almost-duplicate neighbours on both sides: validation accuracy lands above
    95% while measuring nothing. IPD_Project_Guide.md line 421 flags this
    explicitly. Refuse to train rather than report a fabricated number.
    """
    train_x = np.load(data_dir / "train_spectra.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    val_path_x = data_dir / "val_spectra.npy"
    val_path_y = data_dir / "val_labels.npy"
    if val_path_x.is_file() and val_path_y.is_file():
        return train_x, train_y, np.load(val_path_x), np.load(val_path_y)
    missing = [path.name for path in (val_path_x, val_path_y) if not path.is_file()]
    raise FileNotFoundError(
        f"missing validation arrays in {data_dir}: {', '.join(missing)}. "
        "Refusing to fall back to a random pixel split: neighbouring pixels in a "
        "hyperspectral field are near-identical, so that inflates validation "
        "accuracy above ~95% without measuring generalisation. Build a "
        "sample-level split with `sample_level_split()` (grouping by source "
        "sample, not by pixel) and save val_spectra.npy / val_labels.npy first."
    )


def train_loop(
    model: torch.nn.Module,
    config: dict,
    data_dir: Path = DEFAULT_DATA_DIR,
    use_wandb: bool = False,
) -> None:
    train_x, train_y, val_x, val_y = _load_arrays(data_dir)
    classes = train_y.max() + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    counts = np.bincount(train_y, minlength=int(classes)).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(
        counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    patience = int(config["training"].get("early_stopping_patience", 10))
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])

    if use_wandb:
        import wandb

        wandb.init(project="microplastic-ipd", name="spectral-1dcnn", config=config)

    best_f1 = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_x))
        for start in range(0, len(perm), batch_size):
            batch = perm[start : start + batch_size]
            xb = torch.from_numpy(train_x[batch]).float().to(device)
            yb = torch.from_numpy(train_y[batch]).long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(val_x).float().to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(val_y, preds, average="macro"))
        val_loss = float(criterion(logits, torch.from_numpy(val_y).long().to(device)))
        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": float(loss),
                    "val/macro_f1": val_f1,
                    "val/loss": val_loss,
                }
            )
        print(f"epoch {epoch:3d}  loss {float(loss):.4f}  val_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            stale = 0
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": list(POLYMER_CLASSES),
                    "input_bands": int(train_x.shape[1]),
                },
                WEIGHTS_DIR / "spectral_1dcnn_best.pth",
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}")
                break


def train_svm_baseline(config: dict, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    train_x, train_y, val_x, val_y = _load_arrays(data_dir)
    svm = SVC(kernel="rbf", class_weight="balanced")
    svm.fit(train_x, train_y)
    preds = svm.predict(val_x)
    print(f"SVM val macro-F1: {f1_score(val_y, preds, average='macro'):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train_spectral")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--svm-baseline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    set_seed(args.seed)
    model = build_model(config)
    train_loop(model, config, data_dir=args.data, use_wandb=args.wandb)
    if args.svm_baseline:
        train_svm_baseline(config, data_dir=args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
