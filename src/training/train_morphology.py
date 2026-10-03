"""Training loop for the morphology classifier (Pipeline 1B)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence, Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # allow `python src/training/...py` from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.models.vision_tower import VisionTower  # noqa: E402
from src.preprocessing.image_preprocess import (  # noqa: E402
    get_eval_transform,
    get_train_transform,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "training_config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "morphology"

MORPHOLOGY_CLASSES: tuple[str, ...] = ("sphere", "fragment", "fiber", "film", "foam")


def set_seed(seed: int) -> None:
    """Seed torch/numpy/random for reproducible runs (Guide, Section 5)."""
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


def build_model(config: dict) -> VisionTower:
    """Construct the MobileNetV3-Small morphology classifier (AGENT.md 4.2)."""
    classes = config.get("data", {}).get("morphology_classes") or list(
        MORPHOLOGY_CLASSES
    )
    return VisionTower(num_classes=len(classes), pretrained=True)


class MorphologyDataset(torch.utils.data.Dataset):
    """ImageFolder-style reader: ``<data>/<split>/<class>/<image>``."""

    def __init__(self, root: Path, classes: Sequence[str], transform) -> None:
        self.samples: list[tuple[str, int]] = []
        self.transform = transform
        for label, class_name in enumerate(classes):
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(image_path), label))
        if not self.samples:
            raise FileNotFoundError(f"no morphology images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, label


def _subset(dataset: MorphologyDataset, limit: int | None) -> torch.utils.data.Dataset:
    if limit is None:
        return dataset
    return torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))


def build_loaders(
    data_dir: Path,
    classes: Sequence[str],
    batch_size: int,
    img_size: int,
    num_workers: int = 2,
    limit: int | None = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Build train/val/test DataLoaders from ``{train,val,test}/<class>/`` folders."""
    train_dataset = _subset(
        MorphologyDataset(data_dir / "train", classes, get_train_transform(img_size)),
        limit,
    )
    val_dataset = _subset(
        MorphologyDataset(data_dir / "val", classes, get_eval_transform(img_size)),
        limit,
    )
    test_dataset = _subset(
        MorphologyDataset(data_dir / "test", classes, get_eval_transform(img_size)),
        limit,
    )
    # DataLoader kwargs are heterogeneous (int/bool/str), so this needs Any for
    # mypy to accept `**loader_kwargs` against the typed DataLoader signature.
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(cast(Sized, train_dataset)) > batch_size,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader


def load_class_weights(
    data_dir: Path, classes: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Inverse-frequency weights from ``class_weights.json``; uniform fallback."""
    weights_file = data_dir / "class_weights.json"
    if not weights_file.is_file():
        print(f"  [warn] {weights_file} not found — using uniform weights")
        return torch.ones(len(classes), device=device)
    with weights_file.open(encoding="utf-8") as handle:
        weights = json.load(handle)
    return torch.tensor(
        [float(weights[class_name]) for class_name in classes], device=device
    )


def build_optimizer(
    model: VisionTower, config: dict, mode: str
) -> torch.optim.Optimizer:
    """NAdam with the freeze schedule per stage (run1/run2/run3)."""
    morph = config.get("morphology", {})
    head_lr = float(morph.get("head_lr", 1e-3))
    weight_decay = float(config.get("training", {}).get("weight_decay", 1e-4))

    if mode == "run1":  # backbone already frozen by the VisionTower constructor
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.NAdam(trainable, lr=head_lr, weight_decay=weight_decay)

    if mode == "run2":  # differential LR: last-4 blocks + head
        model.unfreeze_last_blocks(int(morph.get("unfreeze_blocks", 4)))
        backbone_lr = float(morph.get("unfreeze_lr", 1e-4))
        return torch.optim.NAdam(
            [
                {
                    "params": [
                        p
                        for _, p in model.backbone.named_parameters()
                        if p.requires_grad
                    ],
                    "lr": backbone_lr,
                },
                {
                    "params": [p for p in model.head.parameters() if p.requires_grad],
                    "lr": head_lr,
                },
            ],
            weight_decay=weight_decay,
        )

    model.unfreeze_all()
    return torch.optim.NAdam(
        model.parameters(),
        lr=float(morph.get("full_finetune_lr", 1e-5)),
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: VisionTower,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """One full training pass; returns (mean loss, accuracy)."""
    model.train()
    total = correct = 0
    running_loss = 0.0
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: VisionTower,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    """Evaluate; returns (mean loss, accuracy, macro-F1, preds, labels)."""
    model.eval()
    total = correct = 0
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    use_amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
        macro_f1,
        all_preds,
        all_labels,
    )


def save_checkpoint(
    model: VisionTower,
    path: Path,
    epoch: int,
    val_macro_f1: float,
    classes: Sequence[str],
) -> None:
    """Save ``model.state_dict`` + metadata (load with ``map_location`` on any device)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_macro_f1": val_macro_f1,
            "classes": list(classes),
        },
        path,
    )


def load_checkpoint(
    model: VisionTower, path: Path, device: torch.device
) -> tuple[int, float]:
    """Load a checkpoint saved by :func:`save_checkpoint`."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("val_macro_f1", 0.0))


def run_stage(
    model: VisionTower,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    config: dict,
    device: torch.device,
    mode: str,
    start_epoch: int,
    end_epoch: int,
    history: dict,
    best_checkpoint: Path,
    stage_checkpoint: Path,
    classes: Sequence[str],
    wandb_run=None,
) -> dict:
    """Train one schedule stage (run1/run2/run3) with cosine LR + early stop."""
    optimizer = build_optimizer(model, config, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=end_epoch - start_epoch + 1
    )
    patience = int(config.get("training", {}).get("early_stopping_patience", 10))
    best_f1 = float(history.get("best_val_macro_f1", 0.0))
    epochs_no_improve = 0

    print(f"=== {mode}: epochs {start_epoch}-{end_epoch} ===")
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        print(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_macro_f1 {val_f1:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "run": mode,
                }
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            save_checkpoint(model, best_checkpoint, epoch, val_f1, classes)
            print(f"  -> best macro-F1 {val_f1:.4f} (saved {best_checkpoint.name})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> early stop at epoch {epoch} (patience {patience})")
                break

    save_checkpoint(model, stage_checkpoint, end_epoch, best_f1, classes)
    history["best_val_macro_f1"] = max(
        best_f1, float(history.get("best_val_macro_f1", 0.0))
    )
    return history


def write_reports(
    model: VisionTower,
    test_loader: torch.utils.data.DataLoader,
    classes: Sequence[str],
    device: torch.device,
    history: dict,
    criterion: nn.Module,
    best_checkpoint: Path,
    confusion_matrix_path: Path,
    report_json_path: Path,
    curves_path: Path,
) -> dict:
    """Final test evaluation — exports the confusion-matrix PNG (Task 1)."""
    if best_checkpoint.is_file():
        load_checkpoint(model, best_checkpoint, device)
        print(f"loaded best checkpoint: {best_checkpoint}")

    _, test_acc, test_macro_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("\n=== TEST REPORT ===")
    print(f"accuracy : {test_acc:.4f}")
    print(f"macro-F1 : {test_macro_f1:.4f}")
    for class_name, score in zip(classes, per_class_f1):
        print(f"  {class_name:>10s}: F1 {score:.4f}")

    report = {
        "accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(test_macro_f1), 4),
        "per_class_f1": {
            class_name: round(float(score), 4)
            for class_name, score in zip(classes, per_class_f1)
        },
        "test_samples": len(y_true),
    }
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("saved", report_json_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Morphology Confusion Matrix")
    fig.tight_layout()
    fig.savefig(confusion_matrix_path, dpi=150)
    plt.close(fig)
    print("saved", confusion_matrix_path)

    if history.get("epoch"):
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["val_macro_f1"], color="tab:orange")
        axes[1].set_title("Val macro-F1")
        axes[1].set_xlabel("epoch")
        fig2.tight_layout()
        fig2.savefig(curves_path, dpi=150)
        plt.close(fig2)
        print("saved", curves_path)

    return report


def train_loop(
    model: VisionTower,
    config: dict,
    *,
    data_dir: Path,
    output_root: Path = REPO_ROOT,
    use_wandb: bool = False,
    limit: int | None = None,
    max_epochs: int | None = None,
    workers: int | None = None,
    seed: int = 42,
) -> dict:
    """Run the three-stage progressive-unfreezing schedule + final test report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    print(f"\ndevice: {device}")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    classes = list(
        config.get("data", {}).get("morphology_classes") or MORPHOLOGY_CLASSES
    )
    img_size = int(config.get("data", {}).get("image_size", 224))
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    num_workers = (
        workers
        if workers is not None
        else int(config.get("training", {}).get("num_workers", 2))
    )

    train_loader, val_loader, test_loader = build_loaders(
        data_dir, classes, batch_size, img_size, num_workers=num_workers, limit=limit
    )
    criterion = nn.CrossEntropyLoss(
        weight=load_class_weights(data_dir, classes, device)
    )
    # CrossEntropyLoss stores `weight` verbatim, so it is never None here.
    print("loss class weights:", criterion.weight.tolist())  # type: ignore[union-attr]

    weights_dir = output_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_root / "results" / "classification"
    best_checkpoint = weights_dir / "mobilenetv3_morphology_best.pth"

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="microplastic-ipd", name="morphology-unfreeze", config=config
        )

    morph = config.get("morphology", {})
    freeze_end = int(morph.get("freeze_epochs", 10))
    unfreeze_end = int(morph.get("unfreeze_epochs", 30))
    full_end = int(morph.get("full_finetune_epochs", 50))
    if max_epochs:
        max_epochs = max(max_epochs, 1)
        freeze_end = min(freeze_end, max_epochs)
        unfreeze_end = min(unfreeze_end, max_epochs)
        full_end = min(full_end, max_epochs)

    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "best_val_macro_f1": 0.0,
    }
    stages = [
        ("run1", 1, freeze_end),
        ("run2", freeze_end + 1, unfreeze_end),
        ("run3", unfreeze_end + 1, full_end),
    ]
    for mode, start, end in stages:
        if start > end:
            continue
        run_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            config,
            device,
            mode,
            start,
            end,
            history,
            best_checkpoint,
            weights_dir / f"{mode}.pth",
            classes,
            wandb_run,
        )

    if wandb_run is not None:
        wandb_run.finish()

    return write_reports(
        model,
        test_loader,
        classes,
        device,
        history,
        criterion,
        best_checkpoint,
        results_dir / "confusion_morphology.png",
        results_dir / "per_class_f1.json",
        results_dir / "training_curves.png",
    )


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
        "--epochs", type=int, default=None, help="Cap the total number of epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)."
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per split (smoke testing on small subsets).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker count (overrides config; use 0 on Windows "
        "so spawned processes do not re-enter the job).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT,
        help="Output root for weights/ and results/ (default: repo root).",
    )
    parser.add_argument(
        "--resnet-baseline",
        action="store_true",
        help="Train a ResNet18 baseline for the H2 McNemar comparison "
        "(deferred to the end-of-project tuning/verification phase).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)

    print(f"config : {args.config}")
    print(f"data   : {args.data}")
    print(f"seed   : {args.seed}   wandb: {args.wandb}   limit: {args.limit}")

    if args.resnet_baseline:
        raise NotImplementedError(
            "--resnet-baseline is scaffolded only: the ResNet18 baseline for H2 is "
            "implemented in the end-of-project tuning/verification phase "
            "(AGENT.md Section 6/8)."
        )

    model = build_model(config)
    train_loop(
        model,
        config,
        data_dir=args.data,
        output_root=args.output,
        use_wandb=args.wandb,
        limit=args.limit,
        max_epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
