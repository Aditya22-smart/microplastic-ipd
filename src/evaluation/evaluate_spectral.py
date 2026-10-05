from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.spectral_tower import SpectralTower


TEST_SPECTRA = Path(
    "data/processed/spectra/c4_dedup_test_spectra.npy"
)

TEST_LABELS = Path(
    "data/processed/spectra/c4_dedup_test_labels.npy"
)

CHECKPOINT = Path(
    "weights/spectral_1dcnn_dedup_best.pth"
)

RESULTS_DIR = Path("results/classification")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONFUSION_PATH = (
    RESULTS_DIR / "confusion_polymer.png"
)

ROC_PATH = (
    RESULTS_DIR / "roc_auc_polymer.png"
)

F1_TABLE_PATH = (
    RESULTS_DIR / "polymer_f1_table.csv"
)

METRICS_PATH = (
    RESULTS_DIR / "spectral_metrics.json"
)


CLASSES = [
    "HDPE",
    "LDPE",
    "PET",
    "PP",
    "PS",
    "PVC",
]

NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 64


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


X_test = np.load(
    TEST_SPECTRA
)

y_test_raw = np.load(
    TEST_LABELS
)

print(
    f"Test spectra shape: {X_test.shape}"
)

print(
    f"Test labels shape: {y_test_raw.shape}"
)


label_to_idx = {
    label: i
    for i, label in enumerate(CLASSES)
}

y_test = np.array(
    [
        label_to_idx[label]
        for label in y_test_raw
    ],
    dtype=np.int64,
)

print(
    f"Classes: {CLASSES}"
)

print(
    f"Test samples: {len(y_test)}"
)


model = SpectralTower(
    num_classes=NUM_CLASSES,
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    f"Loaded checkpoint: {CHECKPOINT}"
)


X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)

y_tensor = torch.tensor(
    y_test,
    dtype=torch.long,
)

dataset = TensorDataset(
    X_tensor,
    y_tensor,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


all_logits = []
all_targets = []


with torch.no_grad():

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)

        logits = model(
            X_batch
        )

        all_logits.append(
            logits.cpu()
        )

        all_targets.append(
            y_batch
        )


logits = torch.cat(
    all_logits
).numpy()

y_true = torch.cat(
    all_targets
).numpy()


exp_logits = np.exp(
    logits
    - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
)

y_prob = (
    exp_logits
    / exp_logits.sum(
        axis=1,
        keepdims=True,
    )
)

y_pred = np.argmax(
    y_prob,
    axis=1,
)


accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
)


print(
    "\n===== TEST RESULTS ====="
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Macro-F1    : {macro_f1:.4f}"
)

print(
    f"Weighted-F1 : {weighted_f1:.4f}"
)


report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(
        NUM_CLASSES
    ),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)


print(
    "\n===== CLASSIFICATION REPORT ====="
)


for class_name in CLASSES:

    row = report[
        class_name
    ]

    print(
        f"{class_name:5s} | "
        f"Precision: {row['precision']:.4f} | "
        f"Recall: {row['recall']:.4f} | "
        f"F1: {row['f1-score']:.4f} | "
        f"Support: {int(row['support'])}"
    )


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(
        NUM_CLASSES
    ),
)


print(
    "\n===== RAW CONFUSION MATRIX ====="
)

print(cm)


row_totals = cm.sum(
    axis=1,
    keepdims=True,
)

cm_percent = (
    cm.astype(float)
    / row_totals
) * 100.0


print(
    "\n===== NORMALIZED CONFUSION MATRIX (%) ====="
)

print(
    np.round(
        cm_percent,
        2,
    )
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    cm_percent,
    vmin=0,
    vmax=100,
)

ax.set_xticks(
    np.arange(NUM_CLASSES)
)

ax.set_yticks(
    np.arange(NUM_CLASSES)
)

ax.set_xticklabels(CLASSES)
ax.set_yticklabels(CLASSES)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "True Label"
)

ax.set_title(
    "Polymer Classification Confusion Matrix (%)"
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            f"{cm_percent[i, j]:.1f}%",
            ha="center",
            va="center",
        )


cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Percentage (%)"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=300,
)

plt.close()


print(
    f"\nSaved normalized confusion matrix: "
    f"{CONFUSION_PATH}"
)


y_true_bin = label_binarize(
    y_true,
    classes=np.arange(
        NUM_CLASSES
    ),
)


try:

    macro_roc_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr",
    )

    weighted_roc_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="weighted",
        multi_class="ovr",
    )

    print(
        f"Macro ROC-AUC    : "
        f"{macro_roc_auc:.4f}"
    )

    print(
        f"Weighted ROC-AUC : "
        f"{weighted_roc_auc:.4f}"
    )

    per_class_auc = {}

    for i, class_name in enumerate(CLASSES):

        per_class_auc[class_name] = (
            roc_auc_score(
                y_true_bin[:, i],
                y_prob[:, i],
            )
        )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    for i, class_name in enumerate(CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_prob[:, i],
        )

        ax.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={per_class_auc[class_name]:.3f})"
            ),
        )


    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Polymer One-vs-Rest ROC Curves"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        ROC_PATH,
        dpi=300,
    )

    plt.close()


    print(
        f"Saved ROC-AUC plot: {ROC_PATH}"
    )


except ValueError as e:

    print(
        f"ROC-AUC could not be calculated: {e}"
    )

    macro_roc_auc = None
    weighted_roc_auc = None
    per_class_auc = {}


f1_rows = []


for class_name in CLASSES:

    f1_rows.append(
        {
            "polymer": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": int(
                report[class_name]["support"]
            ),
        }
    )


f1_df = pd.DataFrame(
    f1_rows
)

f1_df.to_csv(
    F1_TABLE_PATH,
    index=False,
)


print(
    f"Saved F1 table: {F1_TABLE_PATH}"
)


metrics = {
    "dataset": "FTIR-PLASTIC-c4 deduplicated",
    "checkpoint": str(CHECKPOINT),
    "test_samples": int(len(y_true)),
    "classes": CLASSES,
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "macro_roc_auc": (
        None
        if macro_roc_auc is None
        else float(macro_roc_auc)
    ),
    "weighted_roc_auc": (
        None
        if weighted_roc_auc is None
        else float(weighted_roc_auc)
    ),
    "per_class_roc_auc": {
        k: float(v)
        for k, v in per_class_auc.items()
    },
    "confusion_matrix_raw": cm.tolist(),
    "confusion_matrix_percent": (
        cm_percent.tolist()
    ),
    "classification_report": report,
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


print(
    f"Saved metrics JSON: {METRICS_PATH}"
)

print(
    "\n===== DAY 6 EVALUATION COMPLETE ====="
)