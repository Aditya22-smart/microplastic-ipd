import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit


def savitzky_golay_smooth(spectra, window_length=11, polyorder=2):

    spectra = np.asarray(spectra, dtype=np.float32)

    return savgol_filter(
        spectra, window_length=window_length, polyorder=polyorder, axis=1
    ).astype(np.float32)


def zero_bands(spectra, start_band=280, end_band=320):

    spectra = np.asarray(spectra, dtype=np.float32).copy()

    spectra[:, start_band : end_band + 1] = 0.0

    return spectra


def snv_normalize(spectra):

    spectra = np.asarray(spectra, dtype=np.float32)

    row_mean = np.mean(spectra, axis=1, keepdims=True)

    row_std = np.std(spectra, axis=1, keepdims=True)

    row_std = np.where(row_std == 0, 1.0, row_std)

    normalized = (spectra - row_mean) / row_std

    return normalized.astype(np.float32)


def preprocess_spectra(spectra):

    spectra = savitzky_golay_smooth(spectra, window_length=11, polyorder=2)

    spectra = zero_bands(spectra, start_band=280, end_band=320)

    spectra = snv_normalize(spectra)

    return spectra


def sample_level_split(spectra, labels, sample_ids, test_size=0.2, random_state=42):

    spectra = np.asarray(spectra)
    labels = np.asarray(labels)
    sample_ids = np.asarray(sample_ids)

    if len(spectra) != len(labels):
        raise ValueError("spectra and labels must have same length.")

    if len(spectra) != len(sample_ids):
        raise ValueError("spectra and sample_ids must have same length.")

    unique_samples = np.unique(sample_ids)

    if len(unique_samples) < 2:
        raise ValueError(
            "At least 2 unique samples/images are required "
            "for a sample-level train/test split."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(spectra, labels, groups=sample_ids))

    return (
        spectra[train_idx],
        spectra[test_idx],
        labels[train_idx],
        labels[test_idx],
        sample_ids[train_idx],
        sample_ids[test_idx],
    )
