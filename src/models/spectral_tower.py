"""1D-CNN polymer classifier (Pipeline 2B)."""

from __future__ import annotations

import torch
import torch.nn as nn

# C4 polymer classes (6-class deduplicated model)
POLYMER_CLASSES: tuple[str, ...] = ("HDPE", "LDPE", "PET", "PP", "PS", "PVC")
# Legacy Zenodo 5-class set for backward compatibility
ZENODO_CLASSES: tuple[str, ...] = ("PE", "PP", "PS", "PMMA", "PAN")


class SpectralTower(nn.Module):
    """Conv1D network over a single preprocessed spectrum."""

    def __init__(self, num_classes: int = 6, input_bands: int = 3736) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_bands = input_bands

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x
