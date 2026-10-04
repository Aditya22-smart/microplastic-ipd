r"""Vanilla gradient saliency for 1D spectral polymer classifier (Pipeline 2B - Day 8).

Computes input-level gradient saliency:
    S(x) = | \frac{\partial \hat{y}_c}{\partial x} |
to identify which infrared spectral bands (wavenumbers in cm^-1) most strongly
influence the 1D-CNN's classification decisions for each polymer type.

Outputs are saved to results/classification/spectral_saliency_maps/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.spectral_tower import SpectralTower

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "spectral_1dcnn_dedup_best.pth"
DEFAULT_TEST_X = REPO_ROOT / "data" / "processed" / "spectra" / "c4_dedup_test_spectra.npy"
DEFAULT_TEST_Y = REPO_ROOT / "data" / "processed" / "spectra" / "c4_dedup_test_labels.npy"
DEFAULT_C4_CSV = (
    REPO_ROOT
    / "data"
    / "raw"
    / "ftir_plastic_c4"
    / "FTIR-PLASTIC-c4"
    / "FTIR_PLASTIC_c4_deduplicated.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "classification" / "spectral_saliency_maps"

CLASSES: list[str] = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

# Characteristic FTIR absorption bands (wavenumbers in cm^-1)
CHARACTERISTIC_BANDS: dict[str, list[tuple[float, float, str]]] = {
    "HDPE": [
        (2840, 2960, "C-H stretch"),
        (1460, 1475, "CH2 bend / scissor"),
        (715, 735, "CH2 rocking"),
    ],
    "LDPE": [
        (2840, 2960, "C-H stretch"),
        (1460, 1475, "CH2 bend / scissor"),
        (715, 735, "CH2 rocking"),
        (1370, 1380, "CH3 umbrella (branching)"),
    ],
    "PP": [
        (2840, 2960, "C-H stretch"),
        (1450, 1465, "CH3 / CH2 bend"),
        (1370, 1385, "CH3 symmetric bend"),
        (970, 1005, "C-C chain skeletal"),
        (835, 850, "C-CH3 stretch"),
    ],
    "PS": [
        (3000, 3100, "Aromatic C-H stretch"),
        (2850, 2930, "Aliphatic C-H stretch"),
        (1595, 1610, "Aromatic ring C=C"),
        (1485, 1500, "Aromatic ring mode"),
        (1445, 1460, "CH2 bend / ring"),
        (690, 710, "Mono-sub aromatic bend"),
        (745, 765, "Mono-sub aromatic bend"),
    ],
    "PET": [
        (1710, 1735, "Ester C=O stretch"),
        (1235, 1280, "Aromatic ester C-O-C"),
        (1090, 1130, "Aliphatic ester C-O"),
        (720, 735, "Aromatic C-H wag"),
    ],
    "PVC": [
        (2900, 2980, "C-H stretch"),
        (1420, 1440, "CH2 scissor (alpha to Cl)"),
        (1245, 1265, "C-H bend"),
        (600, 695, "C-Cl stretch"),
    ],
}


def load_wavenumber_grid(c4_csv_path: Path = DEFAULT_C4_CSV) -> np.ndarray:
    """Load the 3736 wavenumber values (in cm^-1) from the C4 dataset."""
    if c4_csv_path.is_file():
        df_row = pd.read_csv(c4_csv_path, nrows=1)
        grid_cols = [c for c in df_row.columns if c.startswith("Data(x)")]
        if grid_cols:
            return np.asarray(df_row[grid_cols].iloc[0], dtype=np.float32)
    return np.linspace(399.1927, 4000.6045, 3736, dtype=np.float32)


def compute_spectral_saliency(
    model: SpectralTower,
    spectrum: np.ndarray | torch.Tensor,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute vanilla input gradient saliency |d(logit_c) / d(x)|."""
    if device is None:
        device = next(model.parameters()).device

    if isinstance(spectrum, np.ndarray):
        x = torch.from_numpy(spectrum.astype(np.float32))
    else:
        x = spectrum.clone().float()

    if x.ndim == 1:
        x = x.unsqueeze(0)

    x = x.to(device).requires_grad_(True)
    model.zero_grad()

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(logits.argmax(dim=1).item())
    conf = float(probs[0, pred_idx].item())

    c = pred_idx if target_class is None else target_class
    target_logit = logits[0, c]
    target_logit.backward()

    grad = x.grad[0].detach().cpu().numpy()
    saliency = np.abs(grad)

    s_max = saliency.max()
    saliency_norm = saliency / s_max if s_max > 0 else saliency

    return saliency_norm, pred_idx, conf


def plot_saliency_map(
    wavenumbers: np.ndarray,
    spectrum: np.ndarray,
    saliency: np.ndarray,
    true_label: str,
    pred_label: str,
    confidence: float,
    out_path: Path,
) -> None:
    """Plot the spectrum and aligned saliency map with characteristic bands."""
    fig, (ax_spec, ax_sal) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    # 1. Spectrum plot
    ax_spec.plot(wavenumbers, spectrum, color="#1f77b4", linewidth=1.2, label="Preprocessed Spectrum (SNV)")
    ax_spec.set_ylabel("Intensity", fontsize=10)
    title = f"Polymer: {true_label} (Predicted: {pred_label}, Confidence: {confidence:.2%})"
    ax_spec.set_title(title, fontsize=12, fontweight="bold")
    ax_spec.grid(True, linestyle="--", alpha=0.4)
    ax_spec.legend(loc="upper right", fontsize=9)

    # 2. Saliency map plot
    ax_sal.plot(wavenumbers, saliency, color="#d62728", linewidth=1.2, label=r"Gradient Saliency $|\partial y/\partial x|$")
    ax_sal.fill_between(wavenumbers, 0, saliency, color="#d62728", alpha=0.25)
    ax_sal.set_xlabel(r"Wavenumber ($\mathrm{cm^{-1}}$)", fontsize=10)
    ax_sal.set_ylabel("Normalized Saliency", fontsize=10)
    ax_sal.set_ylim(-0.02, 1.05)
    ax_sal.grid(True, linestyle="--", alpha=0.4)

    # Highlight characteristic bands
    bands = CHARACTERISTIC_BANDS.get(true_label, [])
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for i, (low, high, name) in enumerate(bands):
        c = colors[i % len(colors)]
        ax_sal.axvspan(low, high, color=c, alpha=0.18, label=f"{name} ({int(low)}-{int(high)})")
        ax_spec.axvspan(low, high, color=c, alpha=0.12)

    ax_sal.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_sal.set_xlim(4000, 400)  # Spectroscopy standard convention

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_saliency_maps(
    weights_path: Path = DEFAULT_WEIGHTS,
    test_x_path: Path = DEFAULT_TEST_X,
    test_y_path: Path = DEFAULT_TEST_Y,
    c4_csv_path: Path = DEFAULT_C4_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    samples_per_class: int = 3,
) -> dict:
    """Generate saliency maps for all polymer classes and save summary plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model = SpectralTower(num_classes=len(CLASSES), input_bands=3736)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)
    wavenumbers = load_wavenumber_grid(c4_csv_path)

    results_summary: dict = {
        "dataset": "FTIR-PLASTIC-c4 (Deduplicated)",
        "model": "1D-CNN (SpectralTower)",
        "classes": CLASSES,
        "class_evaluations": {},
    }

    fig_grid, axes = plt.subplots(
        nrows=len(CLASSES),
        ncols=2,
        figsize=(15, 3 * len(CLASSES)),
        sharex=True,
    )

    for row_idx, cls_name in enumerate(CLASSES):
        indices = np.where(test_y == cls_name)[0]
        if len(indices) == 0:
            continue

        class_saliencies = []
        for s_idx in range(min(samples_per_class, len(indices))):
            sample_idx = indices[s_idx]
            spec = test_x[sample_idx]

            sal, pred_idx, conf = compute_spectral_saliency(
                model=model,
                spectrum=spec,
                target_class=CLASSES.index(cls_name),
                device=device,
            )
            class_saliencies.append(sal)

            sample_plot_path = out_dir / f"saliency_{cls_name}_sample_{s_idx + 1}.png"
            plot_saliency_map(
                wavenumbers=wavenumbers,
                spectrum=spec,
                saliency=sal,
                true_label=cls_name,
                pred_label=CLASSES[pred_idx],
                confidence=conf,
                out_path=sample_plot_path,
            )

        mean_spec = test_x[indices].mean(axis=0)
        mean_sal = np.mean(class_saliencies, axis=0)
        mean_sal = mean_sal / (mean_sal.max() + 1e-8)

        ax_sp = axes[row_idx, 0]
        ax_sl = axes[row_idx, 1]

        ax_sp.plot(wavenumbers, mean_spec, color="#1f77b4", linewidth=1.0)
        ax_sp.set_ylabel(f"{cls_name}\nSNV", fontsize=9, fontweight="bold")
        ax_sp.grid(True, linestyle="--", alpha=0.3)

        ax_sl.plot(wavenumbers, mean_sal, color="#d62728", linewidth=1.0)
        ax_sl.fill_between(wavenumbers, 0, mean_sal, color="#d62728", alpha=0.25)
        ax_sl.set_ylabel("Saliency", fontsize=9)
        ax_sl.grid(True, linestyle="--", alpha=0.3)

        bands = CHARACTERISTIC_BANDS.get(cls_name, [])
        for low, high, name in bands:
            ax_sl.axvspan(low, high, color="#2ca02c", alpha=0.18)
            ax_sp.axvspan(low, high, color="#2ca02c", alpha=0.12)

        top_indices = np.argsort(mean_sal)[-5:][::-1]
        top_wavenumbers = [float(wavenumbers[i]) for i in top_indices]

        results_summary["class_evaluations"][cls_name] = {
            "samples_evaluated": len(class_saliencies),
            "top_salient_wavenumbers": top_wavenumbers,
            "characteristic_bands": [
                {"range": [low, high], "assignment": name} for low, high, name in bands
            ],
        }

    axes[-1, 0].set_xlabel(r"Wavenumber ($\mathrm{cm^{-1}}$)", fontsize=10)
    axes[-1, 1].set_xlabel(r"Wavenumber ($\mathrm{cm^{-1}}$)", fontsize=10)
    axes[-1, 0].set_xlim(4000, 400)
    axes[-1, 1].set_xlim(4000, 400)
    axes[0, 0].set_title("Mean Spectrum (SNV)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Mean Vanilla Gradient Saliency", fontsize=11, fontweight="bold")

    fig_grid.suptitle(
        "Day 8: Spectral Vanilla Gradient Saliency Across Polymer Classes",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    summary_plot_path = out_dir / "spectral_saliency_all_classes.png"
    fig_grid.savefig(summary_plot_path, dpi=300)
    plt.close(fig_grid)

    summary_json_path = out_dir / "spectral_saliency_summary.json"
    summary_json_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")

    return results_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral_saliency",
        description="Generate vanilla gradient saliency maps for spectral 1D-CNN.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--test-x", type=Path, default=DEFAULT_TEST_X)
    parser.add_argument("--test-y", type=Path, default=DEFAULT_TEST_Y)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("Generating spectral vanilla gradient saliency maps...")
    generate_all_saliency_maps(
        weights_path=args.weights,
        test_x_path=args.test_x,
        test_y_path=args.test_y,
        out_dir=args.out_dir,
        samples_per_class=args.samples_per_class,
    )
    print("Done! Saliency maps and summary saved to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
