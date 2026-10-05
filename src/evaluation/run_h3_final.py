"""Day 7: H3 Hypothesis Testing — Robust 1D-CNN vs SVM on Out-Of-Distribution FLOPP-e Weathered Spectra.

Hypothesis H3: 1D-CNN polymer classifier generalises to weathered spectra (FLOPP-e)
better than SVM baseline.
Statistical Test: Paired Wilcoxon signed-rank test across 5 random seeds (42, 43, 44, 45, 46).
Decision Rule: p < 0.05, alternative="greater".

Wavelength Alignment:
    Common active ATR-FTIR range: 675 - 4000 cm^-1 (3448 bands).
    Addresses FLOPP-e ATR diamond crystal detector cutoff (<675 cm^-1 zero region).
Target Classes (5-Class IPD Guide Standard):
    PE (HDPE + LDPE), PP, PS, PVC, PET.
    Total FLOPP-e Weathered Test Samples: 158.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 128
EPOCHS = 20
CLASSES = ["PE", "PP", "PS", "PVC", "PET"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load C4 dataset & establish the common valid wavenumber grid (675 - 4000 cm^-1)
C4_PATH = REPO_ROOT / "data" / "raw" / "ftir_plastic_c4" / "FTIR-PLASTIC-c4" / "FTIR_PLASTIC_c4_deduplicated.csv"
c4_df = pd.read_csv(C4_PATH)
x_cols = [c for c in c4_df.columns if c.startswith("Data(x)")]
y_cols = [c for c in c4_df.columns if c.startswith("Data(y)")]
grid = np.asarray(c4_df[x_cols].iloc[0], dtype=float)

valid_mask = (grid >= 675) & (grid <= 4000)
valid_grid = grid[valid_mask]
n_bands = len(valid_grid)

# Map HDPE & LDPE -> PE; keep only the 5 target classes
c4_polymers = c4_df["Polymer"].values
c4_mapped = np.array(["PE" if p in ["HDPE", "LDPE"] else p for p in c4_polymers])
mask_5 = np.isin(c4_mapped, CLASSES)

train_raw = c4_df.loc[mask_5, y_cols].values[:, valid_mask]
train_y_str = c4_mapped[mask_5]
train_groups = c4_df.loc[mask_5, "Sample"].values

# SNV preprocessing on the common valid spectral window
X = (train_raw - train_raw.mean(axis=1, keepdims=True)) / train_raw.std(axis=1, keepdims=True)
y = np.array([CLASS_TO_IDX[c] for c in train_y_str], dtype=np.int64)

# 2. Load FLOPP-e weathered spectra
FLOPP_DIR = REPO_ROOT / "data" / "raw" / "flopp_e" / "FLOPP and FLOPP-e" / "FLOPP-e .csv"
flopp_files = sorted(FLOPP_DIR.glob("*.CSV"))

flopp_spectra = []
flopp_y = []

for f in flopp_files:
    stem = f.stem.upper()
    match = None
    for p in CLASSES:
        if stem.startswith(p + " "):
            match = p
            break
    if not match:
        continue

    d = np.loadtxt(f, delimiter=",")
    f_mask = d[:, 0] >= 675
    res = np.interp(valid_grid, d[f_mask, 0], d[f_mask, 1])
    res_snv = (res - res.mean()) / (res.std() + 1e-8)
    flopp_spectra.append(res_snv)
    flopp_y.append(CLASS_TO_IDX[match])

flopp_X = np.stack(flopp_spectra).astype(np.float32)
flopp_y = np.array(flopp_y, dtype=np.int64)

Path("weights").mkdir(exist_ok=True)
Path("results/hypothesis").mkdir(exist_ok=True)

print(f"C4 Training data : {X.shape} ({len(np.unique(train_groups))} physical samples)")
print(f"FLOPP-e OOD data : {flopp_X.shape} (158 weathered spectra across 5 classes)")
print(f"Spectral bands   : {n_bands} (wavenumbers: {valid_grid.min():.1f} - {valid_grid.max():.1f} cm^-1)")
print(f"Device           : {device}", flush=True)


class RobustSpectralTower(nn.Module):
    def __init__(self, num_classes: int = 5, input_bands: int = 3448):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x).squeeze(-1))


results = []

for seed in SEEDS:
    print(f"\n==================== SEED {seed} ====================", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, train_groups))

    counts = np.bincount(y[train_idx], minlength=len(CLASSES))
    class_weights = len(train_idx) / (len(CLASSES) * np.maximum(counts, 1))

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = RobustSpectralTower(num_classes=len(CLASSES), input_bands=n_bands).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    tx = torch.from_numpy(X[train_idx]).float().to(device)
    ty = torch.from_numpy(y[train_idx]).long().to(device)
    vx = torch.from_numpy(X[val_idx]).float().to(device)
    vy = y[val_idx]

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_idx), device=device)

        for start in range(0, len(train_idx), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            bx = tx[idx]
            # Spectral augmentation: subtle jitter and noise
            noise = torch.randn_like(bx) * 0.02
            shift = np.random.randint(-2, 3)
            bx_aug = torch.roll(bx + noise, shifts=shift, dims=-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx_aug)
            loss = loss_fn(logits, ty[idx])
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(vx).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(vy, val_preds, average="macro", zero_division=0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Val Macro-F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_path = Path(f"weights/h3_seed_{seed}_best.pth")
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate 1D-CNN on FLOPP-e OOD
    with torch.no_grad():
        cnn_preds = model(torch.from_numpy(flopp_X).float().to(device)).argmax(dim=1).cpu().numpy()

    cnn_acc = accuracy_score(flopp_y, cnn_preds)
    cnn_macro_f1 = f1_score(flopp_y, cnn_preds, average="macro", zero_division=0)
    cnn_weighted_f1 = f1_score(flopp_y, cnn_preds, average="weighted", zero_division=0)

    # Train & evaluate RBF-SVM on identical split
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    svm.fit(X[train_idx], y[train_idx])
    svm_preds = svm.predict(flopp_X)

    svm_acc = accuracy_score(flopp_y, svm_preds)
    svm_macro_f1 = f1_score(flopp_y, svm_preds, average="macro", zero_division=0)
    svm_weighted_f1 = f1_score(flopp_y, svm_preds, average="weighted", zero_division=0)

    print(f"Seed {seed} Completed:")
    print(f"  1D-CNN  -> Accuracy: {cnn_acc:.2%}, Macro-F1: {cnn_macro_f1:.4f}")
    print(f"  RBF-SVM -> Accuracy: {svm_acc:.2%}, Macro-F1: {svm_macro_f1:.4f}", flush=True)

    results.append({
        "seed": seed,
        "val_macro_f1": float(best_val_f1),
        "cnn_flopp_accuracy": float(cnn_acc),
        "cnn_flopp_macro_f1": float(cnn_macro_f1),
        "cnn_flopp_weighted_f1": float(cnn_weighted_f1),
        "svm_flopp_accuracy": float(svm_acc),
        "svm_flopp_macro_f1": float(svm_macro_f1),
        "svm_flopp_weighted_f1": float(svm_weighted_f1),
    })

    output_5seeds = Path("results/hypothesis/h3_flopp_e_5seeds.json")
    output_5seeds.write_text(json.dumps(results, indent=2), encoding="utf-8")

# 3. Wilcoxon signed-rank test across all 5 seeds
cnn_f1s = [r["cnn_flopp_macro_f1"] for r in results]
svm_f1s = [r["svm_flopp_macro_f1"] for r in results]
cnn_accs = [r["cnn_flopp_accuracy"] for r in results]
svm_accs = [r["svm_flopp_accuracy"] for r in results]

diff = np.asarray(cnn_f1s) - np.asarray(svm_f1s)
d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

try:
    stat_greater, p_greater = wilcoxon(cnn_f1s, svm_f1s, alternative="greater")
    stat_two_sided, p_two_sided = wilcoxon(cnn_f1s, svm_f1s, alternative="two-sided")
except Exception as err:
    stat_greater, p_greater = 0.0, 1.0
    stat_two_sided, p_two_sided = 0.0, 1.0

h3_full_results = {
    "hypothesis": "H3: 1D-CNN polymer classifier generalises to weathered spectra (FLOPP-e) better than SVM",
    "dataset": "FLOPP-e (weathered ATR-FTIR, Rochman Lab)",
    "spectral_range": "675 - 4000 cm^-1 (common valid active window, 3448 bands)",
    "target_classes": CLASSES,
    "num_samples": len(flopp_y),
    "seeds": SEEDS,
    "seed_runs": results,
    "cnn_macro_f1_scores": cnn_f1s,
    "svm_macro_f1_scores": svm_f1s,
    "mean_cnn_macro_f1": float(np.mean(cnn_f1s)),
    "mean_svm_macro_f1": float(np.mean(svm_f1s)),
    "mean_cnn_accuracy": float(np.mean(cnn_accs)),
    "mean_svm_accuracy": float(np.mean(svm_accs)),
    "wilcoxon_test": {
        "test": "Wilcoxon signed-rank",
        "statistic": float(stat_greater),
        "p_value_one_sided_greater": float(p_greater),
        "p_value_two_sided": float(p_two_sided),
        "cohens_d": d,
        "decision": bool(p_greater < 0.05),
    },
    "ocean_raman_status": {
        "dataset": "Ocean Raman Library (Miller 2022, 22 weathered spectra)",
        "status": "Dataset not available in data/raw/ directory. Not evaluated to avoid synthetic data fabrication.",
    },
}

results_path = Path("results/hypothesis/h3_test_results.json")
results_path.write_text(json.dumps(h3_full_results, indent=2), encoding="utf-8")

print("\n==================== H3 STATISTICAL VALIDATION SUMMARY ====================")
print(f"Mean 1D-CNN Accuracy on FLOPP-e  : {np.mean(cnn_accs):.2%}")
print(f"Mean 1D-CNN Macro-F1 on FLOPP-e  : {np.mean(cnn_f1s):.4f}")
print(f"Mean RBF-SVM Accuracy on FLOPP-e : {np.mean(svm_accs):.2%}")
print(f"Mean RBF-SVM Macro-F1 on FLOPP-e : {np.mean(svm_f1s):.4f}")
print(f"Wilcoxon statistic               : {stat_greater}")
print(f"p-value (one-sided greater)      : {p_greater:.5f}")
print(f"Cohen's d                        : {d:.4f}")
print(f"Decision (p < 0.05)              : {p_greater < 0.05}")
print(f"\nFinal report saved to {results_path}", flush=True)
