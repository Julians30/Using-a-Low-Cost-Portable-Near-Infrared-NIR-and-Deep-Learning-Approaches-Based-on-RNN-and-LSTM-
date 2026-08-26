"""Leakage-safe spectral preprocessing utilities.

All fitted quantities must be estimated from the current training partition
only.  This module mirrors the preprocessing logic used in the frozen
notebooks; it does not change the manuscript's model-selection protocol.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


class SpectralPreprocessor:
    """Apply one frozen spectral transform followed by train-fitted scaling.

    Supported methods
    -----------------
    raw
        No spectral transform before standardization.
    snv
        Standard normal variate, applied spectrum-wise.
    msc
        Multiplicative scatter correction using a reference spectrum computed
        from the training partition only.
    sg_smooth
        Savitzky–Golay smoothing in physical wavelength order.
    sg_deriv1
        First Savitzky–Golay derivative in physical wavelength order.
    """

    def __init__(self, method: str, sg_window: int = 11, sg_polyorder: int = 2):
        self.method = str(method).lower()
        self.sg_window = int(sg_window)
        self.sg_polyorder = int(sg_polyorder)
        self.msc_reference_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_spectra, n_wavelengths).")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        return X

    def _base_fit(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        if self.method == "msc":
            self.msc_reference_ = X.mean(axis=0)
        return self._base_transform(X)

    def _base_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)

        if self.method in {"raw", "none"}:
            return X.copy()

        if self.method == "snv":
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd = np.where(sd == 0.0, 1.0, sd)
            return (X - mu) / sd

        if self.method == "msc":
            if self.msc_reference_ is None:
                raise RuntimeError("MSC reference has not been fitted.")
            ref = self.msc_reference_
            ref_centered = ref - ref.mean()
            ref_ss = np.sum(ref_centered**2)
            if ref_ss <= 0:
                raise RuntimeError("Degenerate MSC reference spectrum.")
            x_mean = X.mean(axis=1, keepdims=True)
            slopes = np.sum((X - x_mean) * ref_centered[None, :], axis=1) / ref_ss
            slopes = np.where(np.abs(slopes) < 1e-12, 1.0, slopes)
            intercepts = X.mean(axis=1) - slopes * ref.mean()
            return (X - intercepts[:, None]) / slopes[:, None]

        if self.method == "sg_smooth":
            return savgol_filter(
                X,
                window_length=self.sg_window,
                polyorder=self.sg_polyorder,
                deriv=0,
                axis=1,
                mode="interp",
            )

        if self.method == "sg_deriv1":
            return savgol_filter(
                X,
                window_length=self.sg_window,
                polyorder=self.sg_polyorder,
                deriv=1,
                delta=1.0,
                axis=1,
                mode="interp",
            )

        raise ValueError(f"Unknown preprocessing method: {self.method!r}")

    def fit(self, X: np.ndarray) -> "SpectralPreprocessor":
        X_base = self._base_fit(X)
        self.scaler_ = StandardScaler().fit(X_base)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        X_base = self._base_transform(X)
        return self.scaler_.transform(X_base)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
