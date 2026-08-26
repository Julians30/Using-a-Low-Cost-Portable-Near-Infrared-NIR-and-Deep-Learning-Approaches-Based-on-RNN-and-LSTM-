"""Regression metrics used in the NIR-HUEVOS analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the frozen descriptive regression metrics.

    MAPE is intentionally excluded because the target contains storage day 0.
    """

    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if y.size == 0:
        raise ValueError("Empty inputs are not valid.")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Inputs contain non-finite values.")

    error = p - y
    abs_error = np.abs(error)

    return {
        "MAE_days": float(mean_absolute_error(y, p)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "bias_days": float(np.mean(error)),
        "median_AE_days": float(np.median(abs_error)),
        "within_1d_pct": float(np.mean(abs_error <= 1.0) * 100.0),
        "within_2d_pct": float(np.mean(abs_error <= 2.0) * 100.0),
        "within_3d_pct": float(np.mean(abs_error <= 3.0) * 100.0),
    }


def per_egg_mae(frame, *, sample_col="sample", target_col="storage_days", pred_col="y_pred"):
    """Compute MAE for each egg across its repeated storage-day observations."""

    required = {sample_col, target_col, pred_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["_abs_error"] = np.abs(out[pred_col].astype(float) - out[target_col].astype(float))
    return (
        out.groupby(sample_col, as_index=False)["_abs_error"]
        .mean()
        .rename(columns={"_abs_error": "MAE_days"})
    )
