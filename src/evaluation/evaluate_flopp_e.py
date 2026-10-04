import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC

from src.models.spectral_tower import SpectralTower
from src.preprocessing.spectral_preprocess import preprocess_spectra


FLOPP_DIR = Path("data/raw/flopp_e/FLOPP and FLOPP-e/FLOPP-e .csv")
C4_CSV = Path(
    "data/raw/ftir_plastic_c4/FTIR-PLASTIC-c4/"
    "FTIR_PLASTIC_c4_deduplicated.csv"
)

TRAIN_SPECTRA = Path(
    "data/processed/spectra/c4_dedup_train_spectra.npy"
)
TRAIN_LABELS = Path(
    "data/processed/spectra/c4_dedup_train_labels.npy"
)

CHECKPOINT = Path("weights/spectral_1dcnn_dedup_best.pth")

OUTPUT_DIR = Path("results/hypothesis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ("HDPE", "LDPE", "PET", "PP", "PS", "PVC")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

# FLOPP-e polymers that have an unambiguous matching class
SUPPORTED = {"PET", "PP", "PS", "PVC"}


def get_c4_grid():
    df = pd.read_csv(C4_CSV, nrows=1)

    xcols = [c for c in df.columns if c.startswith("Data(x)")]

    if len(xcols) != 3736:
        raise RuntimeError(
            f"Expected 3736 C4 wavelength points, got {len(xcols)}."
        )

    return np.asarray(df.iloc[0][xcols], dtype=np.float64)


def polymer_from_filename(filename):
    name = Path(filename).stem.upper()

    prefixes = (
        "PET ",
        "PP ",
        "PS ",
        "PVC ",
        "PE ",
        "ABS ",
        "CA ",
        "EAA ",
        "EVA ",
        "PA ",
        "PC ",
        "PMMA ",
        "PU ",
        "SR ",
    )

    if name.startswith("PET "):
        return "PET"
    if name.startswith("PP "):
        return "PP"
    if name.startswith("PS "):
        return "PS"
    if name.startswith("PVC "):
        return "PVC"
    if name.startswith("PE "):
        return "PE"

    if name.startswith("ABS "):
        return "ABS"
    if name.startswith("CA "):
        return "CA"
    if name.startswith("EAA "):
        return "EAA"
    if name.startswith("EVA "):
        return "EVA"
    if name.startswith("PA "):
        return "PA"
    if name.startswith("PC "):
        return "PC"
    if name.startswith("PMMA "):
        return "PMMA"
    if name.startswith("PU "):
        return "PU"
    if name.startswith("SR "):
        return "SR"

    if name.startswith("C"):
        return "Cellulose"

    return "Other"


def load_flopp_e(c4_grid):
    files = sorted(FLOPP_DIR.glob("*.CSV"))

    if len(files) != 195:
        raise RuntimeError(
            f"Expected 195 FLOPP-e spectra, found {len(files)}."
        )

    spectra = []
    labels = []
    filenames = []
    all_classes = []

    for path in files:
        data = np.loadtxt(path, delimiter=",")

        source_grid = data[:, 0].astype(np.float64)
        intensity = data[:, 1].astype(np.float32)

        polymer = polymer_from_filename(path.name)
        all_classes.append(polymer)

        # FLOPP-e -> C4 3736-point grid
        resampled = np.interp(
            c4_grid,
            source_grid,
            intensity,
        ).astype(np.float32)

        if polymer in SUPPORTED:
            spectra.append(resampled)
            labels.append(CLASS_TO_INDEX[polymer])
            filenames.append(path.name)

    return (
        np.stack(spectra),
        np.asarray(labels, dtype=np.int64),
        filenames,
        all_classes,
    )


def evaluate_cnn(X, y, device):
    model = SpectralTower(num_classes=len(CLASSES)).to(device)

    state = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(X).float().to(device)
        logits = model(tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


def evaluate_svm(X, y):
    train_x = np.load(TRAIN_SPECTRA)
    train_labels = np.load(TRAIN_LABELS)

    train_y = np.array(
        [CLASS_TO_INDEX[str(label)] for label in train_labels],
        dtype=np.int64,
    )

    svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    )

    svm.fit(train_x, train_y)

    return svm.predict(X)


def metrics(y, predictions):
    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y,
            predictions,
            labels=sorted(np.unique(y)),
            target_names=[
                CLASSES[i]
                for i in sorted(np.unique(y))
            ],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    print("===== FLOPP-e OOD EVALUATION =====")

    c4_grid = get_c4_grid()

    print(
        "C4 grid:",
        len(c4_grid),
        f"{c4_grid.min():.4f} -> {c4_grid.max():.4f}",
    )

    X_raw, y, filenames, all_classes = load_flopp_e(
        c4_grid
    )

    counts = {}
    for polymer in all_classes:
        counts[polymer] = counts.get(polymer, 0) + 1

    print("Total FLOPP-e spectra:", len(all_classes))
    print("All polymer counts:", counts)
    print("Supported spectra:", len(y))

    # Exact same preprocessing used during C4 training.
    X = preprocess_spectra(X_raw)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    cnn_predictions = evaluate_cnn(
        X,
        y,
        device,
    )

    svm_predictions = evaluate_svm(
        X,
        y,
    )

    cnn_result = metrics(y, cnn_predictions)
    svm_result = metrics(y, svm_predictions)

    result = {
        "dataset": "FLOPP-e",
        "total_spectra": len(all_classes),
        "supported_spectra": len(y),
        "unsupported_spectra": len(all_classes) - len(y),
        "all_polymer_counts": counts,
        "evaluated_classes": [
            CLASSES[i]
            for i in sorted(np.unique(y))
        ],
        "preprocessing": [
            "Savitzky-Golay window=11 polyorder=2",
            "zero bands 280-320",
            "SNV",
        ],
        "resampling": (
            "linear interpolation from FLOPP-e grid "
            "to C4 3736-point grid"
        ),
        "cnn": cnn_result,
        "svm": svm_result,
    }

    output = OUTPUT_DIR / "flopp_e_ood_results.json"

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== CNN =====")
    print(f"Accuracy    : {cnn_result['accuracy']:.4f}")
    print(f"Macro-F1    : {cnn_result['macro_f1']:.4f}")
    print(f"Weighted-F1 : {cnn_result['weighted_f1']:.4f}")

    print()
    print("===== SVM =====")
    print(f"Accuracy    : {svm_result['accuracy']:.4f}")
    print(f"Macro-F1    : {svm_result['macro_f1']:.4f}")
    print(f"Weighted-F1 : {svm_result['weighted_f1']:.4f}")

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
