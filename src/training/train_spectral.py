import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import wandb
from src.models.spectral_tower import SpectralTower

TRAIN_SPECTRA = (
    "data/processed/spectra/c4_dedup_train_spectra.npy"
)
TRAIN_LABELS = (
    "data/processed/spectra/c4_dedup_train_labels.npy"
)
TRAIN_GROUPS = (
    "data/processed/spectra/c4_dedup_train_groups.npy"
)

TEST_SPECTRA = (
    "data/processed/spectra/c4_dedup_test_spectra.npy"
)
TEST_LABELS = (
    "data/processed/spectra/c4_dedup_test_labels.npy"
)
TEST_GROUPS = (
    "data/processed/spectra/c4_dedup_test_groups.npy"
)

CHECKPOINT = (
    "weights/spectral_1dcnn_dedup_best.pth"
)

CLASSES = (
    "HDPE",
    "LDPE",
    "PET",
    "PP",
    "PS",
    "PVC",
)

CLASS_TO_INDEX = {
    name: i
    for i, name in enumerate(CLASSES)
}

SEED = 42

EPOCHS = 50
BATCH_SIZE = 32

INITIAL_LR = 1e-3
ETA_MIN = 1e-6
T_MAX = 50

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

VAL_SIZE = 0.20

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():

    train_x = np.load(TRAIN_SPECTRA)
    train_labels = np.load(TRAIN_LABELS)
    train_groups = np.load(TRAIN_GROUPS)

    test_x = np.load(TEST_SPECTRA)
    test_labels = np.load(TEST_LABELS)
    test_groups = np.load(TEST_GROUPS)

    print("Train spectra:", train_x.shape)
    print("Train labels:", train_labels.shape)
    print(
        "Train samples:",
        len(np.unique(train_groups)),
    )

    print("Test spectra:", test_x.shape)
    print("Test labels:", test_labels.shape)
    print(
        "Test samples:",
        len(np.unique(test_groups)),
    )

    print("Classes:", CLASSES)

    train_test_group_overlap = (
        set(train_groups) & set(test_groups)
    )

    print(
        "Train/Test sample overlap:",
        len(train_test_group_overlap),
    )

    if len(train_test_group_overlap) != 0:
        raise RuntimeError(
            "Data leakage detected: "
            "train/test sample groups overlap."
        )

    train_hashes = {
        row.tobytes()
        for row in train_x
    }

    duplicate_count = sum(
        row.tobytes() in train_hashes
        for row in test_x
    )

    print(
        "Exact train/test spectrum duplicates:",
        duplicate_count,
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Data leakage detected: "
            "exact train/test spectra overlap."
        )

    return (
        train_x,
        train_labels,
        train_groups,
        test_x,
        test_labels,
        test_groups,
    )

def encode_labels(labels):

    return np.array(
        [
            CLASS_TO_INDEX[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )

def make_train_val_split(
    X,
    y,
    groups,
):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    return (
        X[train_idx],
        X[val_idx],
        y[train_idx],
        y[val_idx],
        groups[train_idx],
        groups[val_idx],
    )

def make_loaders(
    train_x,
    train_y,
    val_x,
    val_y,
):

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).long(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_x).float(),
        torch.from_numpy(val_y).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader

def calculate_class_weights(labels):

    counts = np.bincount(
        labels,
        minlength=len(CLASSES),
    )

    weights = len(labels) / (
        len(CLASSES)
        * np.maximum(counts, 1)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            total_loss += (
                loss.item()
                * x.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
    )

def train_model(
    train_loader,
    val_loader,
    class_weights,
    device,
):

    model = SpectralTower(
        num_classes=len(CLASSES)
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_MAX,
            eta_min=ETA_MIN,
        )
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    os.makedirs(
        os.path.dirname(CHECKPOINT),
        exist_ok=True,
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * x.size(0)
            )

        scheduler.step()

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        (
            val_loss,
            val_accuracy,
            val_f1,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"lr={current_lr:.7f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
                "learning_rate": current_lr,
            }
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT,
            )

            print(
                "  Saved best checkpoint -> "
                f"{CHECKPOINT}"
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    print()
    print(
        "Best validation macro-F1:",
        best_f1,
    )
    print(
        "Best epoch:",
        best_epoch,
    )

    return best_f1

def train_svm(
    train_x,
    train_y,
    val_x,
    val_y,
):

    print()
    print("Training SVM baseline...")

    svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=SEED,
    )

    svm.fit(
        train_x,
        train_y,
    )

    predictions = svm.predict(
        val_x
    )

    accuracy = accuracy_score(
        val_y,
        predictions,
    )

    macro_f1 = f1_score(
        val_y,
        predictions,
        average="macro",
        zero_division=0,
    )

    print(
        f"SVM val accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"SVM val macro-F1: "
        f"{macro_f1:.4f}"
    )

    wandb.log(
        {
            "svm_val_accuracy": accuracy,
            "svm_val_macro_f1": macro_f1,
        }
    )

    return svm

def evaluate_test_set(
    model,
    test_x,
    test_y,
    device,
):

    test_dataset = TensorDataset(
        torch.from_numpy(test_x).float(),
        torch.from_numpy(test_y).long(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    (
        loss,
        accuracy,
        macro_f1,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print(
        "FINAL HELD-OUT "
        "DUPLICATE-FREE TEST RESULTS"
    )
    print("--------------------------------")

    print(
        f"Test loss:      {loss:.4f}"
    )

    print(
        f"Test accuracy:  {accuracy:.4f}"
    )

    print(
        f"Test macro-F1:  {macro_f1:.4f}"
    )

    wandb.log(
        {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
        }
    )

    return (
        loss,
        accuracy,
        macro_f1,
    )

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    wandb.init(
        project="microplastic-ipd",
        name="c4-spectral-1dcnn-deduplicated-seed-42",
        config={
            "dataset": "FTIR-PLASTIC-c4",
            "dataset_version": "deduplicated",
            "classes": list(CLASSES),
            "input_bands": 3736,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "NAdam",
            "learning_rate": INITIAL_LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": T_MAX,
            "scheduler_eta_min": ETA_MIN,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "seed": SEED,
            "preprocessing": [
                "Savitzky-Golay "
                "window=11 polyorder=2",
                "zero bands 280-320",
                "SNV",
            ],
            "split": "sample-level",
            "duplicate_handling": (
                "exact spectral duplicates "
                "removed before split"
            ),
        },
    )

    (
        train_x,
        train_labels,
        train_groups,
        test_x,
        test_labels,
        test_groups,
    ) = load_data()

    train_y = encode_labels(
        train_labels
    )

    test_y = encode_labels(
        test_labels
    )

    (
        cnn_train_x,
        cnn_val_x,
        cnn_train_y,
        cnn_val_y,
        cnn_train_groups,
        cnn_val_groups,
    ) = make_train_val_split(
        train_x,
        train_y,
        train_groups,
    )

    print()

    print(
        "CNN train:",
        cnn_train_x.shape,
    )

    print(
        "CNN validation:",
        cnn_val_x.shape,
    )

    print(
        "CNN train samples:",
        len(
            np.unique(
                cnn_train_groups
            )
        ),
    )

    print(
        "CNN validation samples:",
        len(
            np.unique(
                cnn_val_groups
            )
        ),
    )

    overlap = (
        set(cnn_train_groups)
        & set(cnn_val_groups)
    )

    print(
        "Train/validation sample overlap:",
        len(overlap),
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Data leakage detected: "
            "train/validation sample groups overlap."
        )

    (
        train_loader,
        val_loader,
    ) = make_loaders(
        cnn_train_x,
        cnn_train_y,
        cnn_val_x,
        cnn_val_y,
    )

    class_weights = (
        calculate_class_weights(
            cnn_train_y
        )
    )

    print(
        "Class weights:",
        class_weights.numpy(),
    )

    best_f1 = train_model(
        train_loader,
        val_loader,
        class_weights,
        device,
    )

    model = SpectralTower(
        num_classes=len(CLASSES)
    ).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
        )
    )

    model.eval()

    svm = train_svm(
        cnn_train_x,
        cnn_train_y,
        cnn_val_x,
        cnn_val_y,
    )

    evaluate_test_set(
        model,
        test_x,
        test_y,
        device,
    )

    wandb.log(
        {
            "best_val_macro_f1": best_f1,
        }
    )

    wandb.finish()

    print()
    print(
        "C4 deduplicated spectral "
        "training complete."
    )


if __name__ == "__main__":
    main()