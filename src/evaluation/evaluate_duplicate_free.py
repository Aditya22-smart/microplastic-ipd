import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.models.spectral_tower import SpectralTower


p = r"data\processed\spectra"

Xtr = np.load(p + r"\c4_train_spectra.npy")
Xte = np.load(p + r"\c4_test_spectra.npy")
yte = np.load(p + r"\c4_test_labels.npy")

classes = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

train_set = {row.tobytes() for row in Xtr}

keep = np.array(
    [row.tobytes() not in train_set for row in Xte],
    dtype=bool,
)

model = SpectralTower(num_classes=6)
model.load_state_dict(
    torch.load(
        r"weights\spectral_1dcnn_best.pth",
        map_location="cpu",
    )
)
model.eval()

with torch.no_grad():
    logits = model(torch.from_numpy(Xte).float())
    pred = torch.argmax(logits, dim=1).numpy()

y = np.array(
    [classes.index(str(label)) for label in yte],
    dtype=np.int64,
)

print("Original test rows:", len(y))
print("Duplicate rows removed:", int(np.sum(~keep)))
print("Duplicate-free test rows:", int(np.sum(keep)))

print(
    "Accuracy:",
    accuracy_score(y[keep], pred[keep]),
)

print(
    "Macro-F1:",
    f1_score(
        y[keep],
        pred[keep],
        average="macro",
        zero_division=0,
    ),
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y[keep],
        pred[keep],
        target_names=classes,
        zero_division=0,
    )
)
