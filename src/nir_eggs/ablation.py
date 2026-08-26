"""Wavelength-order ablation utilities matching frozen NB05."""

from __future__ import annotations

import numpy as np

SHUFFLE_SEED = 52026
ORDER_CONDITIONS = ["original", "reversed", "shuffled"]


def fixed_shuffled_indices(n_features: int = 331, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Return the target-independent fixed permutation used for NB05."""
    rng = np.random.default_rng(seed)
    return rng.permutation(int(n_features))


def reorder_after_preprocessing(X, condition: str, *, shuffled_indices=None):
    """Reorder already-preprocessed features.

    Critical protocol rule: smoothing/derivatives/MSC/SNV and train-fitted
    scaling are performed in true physical wavelength order first. Only then
    are the transformed features reversed or shuffled. This prevents the
    ablation from changing the mathematical meaning of Savitzky–Golay
    preprocessing.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-D: spectra × wavelengths")
    if condition == "original":
        return X.copy()
    if condition == "reversed":
        return X[:, ::-1].copy()
    if condition == "shuffled":
        idx = fixed_shuffled_indices(X.shape[1]) if shuffled_indices is None else np.asarray(shuffled_indices)
        if sorted(idx.tolist()) != list(range(X.shape[1])):
            raise ValueError("shuffled_indices must be a complete permutation")
        return X[:, idx].copy()
    raise ValueError(f"Unknown order condition: {condition!r}")
