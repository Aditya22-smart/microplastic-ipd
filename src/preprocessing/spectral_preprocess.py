"""Spectral preprocessing pipeline (Pipeline 2A).

Mandatory order: Savitzky-Golay smoothing -> water-vapour band removal -> SNV.
Splits are always by sample/image ID, never by pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit


def savitzky_golay_smooth(spectra, window_length=11, polyorder=2):
    spectra = np.asarray(spectra, dtype=np.float32)
    return savgol_filter(
        spectra, window_length=window_length, polyorder=polyorder, axis=1
    ).astype(np.float32)


def band_mask(wavenumbers, start_cm=280.0, end_cm=320.0):
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    return np.logical_and(wavenumbers >= start_cm, wavenumbers <= end_cm)


def zero_bands(spectra, start_band=280, end_band=320, wavenumbers=None):
    spectra = np.asarray(spectra, dtype=np.float32).copy()
    if wavenumbers is None:
        spectra[:, start_band : end_band + 1] = 0.0
    else:
        spectra[:, band_mask(wavenumbers, start_band, end_band)] = 0.0
    return spectra


def snv_normalize(spectra):
    spectra = np.asarray(spectra, dtype=np.float32)
    row_mean = np.mean(spectra, axis=1, keepdims=True)
    row_std = np.std(spectra, axis=1, keepdims=True)
    row_std = np.where(row_std == 0, 1.0, row_std)
    return ((spectra - row_mean) / row_std).astype(np.float32)


def preprocess_spectra(spectra, wavenumbers=None):
    spectra = savitzky_golay_smooth(spectra, window_length=11, polyorder=2)
    spectra = zero_bands(spectra, start_band=280, end_band=320, wavenumbers=wavenumbers)
    spectra = snv_normalize(spectra)
    return spectra


def sample_level_split(spectra, labels, sample_ids, test_size=0.2, random_state=42):
    spectra = np.asarray(spectra)
    labels = np.asarray(labels)
    sample_ids = np.asarray(sample_ids)

    if len(spectra) != len(labels) or len(spectra) != len(sample_ids):
        raise ValueError("spectra, labels and sample_ids must have the same length.")

    unique_samples = np.unique(sample_ids)
    if len(unique_samples) < 2:
        raise ValueError(
            "At least 2 unique samples/images are required for a sample-level train/test split."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(spectra, labels, groups=sample_ids))

    return (
        spectra[train_idx],
        spectra[test_idx],
        labels[train_idx],
        labels[test_idx],
        sample_ids[train_idx],
        sample_ids[test_idx],
    )


def load_c4_csv(path):
    """Load the FTIR-PLASTIC-c4 wide CSV into (spectra, wavenumbers, labels, class_names)."""
    path = Path(path)
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c.startswith("Data(x)")]
    y_cols = [c for c in df.columns if c.startswith("Data(y)")]
    wavenumbers = df[x_cols].iloc[0].to_numpy(dtype=np.float32) if x_cols else None
    spectra = df[y_cols].to_numpy(dtype=np.float32)
    label_col = next((c for c in df.columns if c.lower() in {"label", "class", "polymer"}), None)
    labels = df[label_col].to_numpy() if label_col else np.zeros(len(spectra), dtype=int)
    class_names = sorted(np.unique(labels).tolist())
    return spectra, wavenumbers, labels, class_names


def parse_envi_scll(path, keep_labels=("PE", "PP", "PS", "PMMA", "PAN")):
    """Parse an ENVI .scll label sidecar into a DataFrame with x/y/class_nr/label."""
    text = Path(path).read_text(encoding="latin1", errors="ignore")
    rows = []
    current = {}
    for line in text.splitlines():
        stripped = line.strip()
        if "#iscEndOfItem" in stripped or "EndOfItem" in stripped:
            if current:
                rows.append(current)
            current = {}
            continue
        if stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            current[key.strip()] = value.strip()
    if current:
        rows.append(current)
    records = []
    for row in rows:
        label = row.get("#iscCaption") or row.get("label") or row.get("name", "")
        class_nr = row.get("class_nr") or row.get("#iscId") or ""
        try:
            x = int(row.get("x", 0))
            y = int(row.get("y", 0))
        except ValueError:
            continue
        if label in keep_labels:
            records.append({"x": x, "y": y, "class_nr": class_nr, "label": label})
    return pd.DataFrame(records, columns=["x", "y", "class_nr", "label"])


def load_hsi_spectra(hdr_path, labels_df):
    """Open an ENVI hyperspectral cube and return (spectra[N, n_bands], wavenumbers)."""
    from spectral import open_image

    img = open_image(str(hdr_path))
    cube = img.load()
    spectra = []
    for _, row in labels_df.iterrows():
        spectra.append(np.squeeze(cube[int(row["y"]), int(row["x"]), :]))
    wavenumbers = np.asarray(img.metadata["wavelength"], dtype=np.float32)
    return np.asarray(spectra, dtype=np.float32), wavenumbers
