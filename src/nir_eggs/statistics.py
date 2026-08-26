"""Egg-level statistical utilities matching the frozen NB06 protocol."""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values):
    """Holm step-down adjustment for a family of p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values in their original comparison order.

    Returns
    -------
    numpy.ndarray
        Holm-adjusted p-values in the same order as the input.
    """

    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p_values must be finite values in [0, 1].")

    m = len(p)
    order = np.argsort(p)
    ordered = p[order]
    adjusted_ordered = np.empty(m, dtype=float)

    running_max = 0.0
    for i, p_i in enumerate(ordered):
        candidate = (m - i) * p_i
        running_max = max(running_max, candidate)
        adjusted_ordered[i] = min(1.0, running_max)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def paired_bootstrap_mean_difference(
    a,
    b,
    *,
    n_resamples: int = 10_000,
    seed: int = 62_026,
    confidence: float = 0.95,
):
    """Paired cluster bootstrap of the mean difference A−B across eggs.

    Each array must contain one inferential value per egg.  Eggs are resampled
    with replacement, preserving the paired model values within each selected
    egg.  This mirrors the cluster/paired bootstrap logic used in NB06.
    """

    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shapes.")
    if a.size < 2:
        raise ValueError("At least two paired eggs are required.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Inputs contain non-finite values.")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1).")

    d = a - b
    rng = np.random.default_rng(seed)
    n = d.size
    boot = np.empty(int(n_resamples), dtype=float)

    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(d[idx])

    alpha = 1.0 - confidence
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "mean_difference": float(np.mean(d)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": int(n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def kendalls_w_from_friedman(chi2: float, n_blocks: int, k_treatments: int) -> float:
    """Compute Kendall's W from a Friedman chi-square statistic."""

    if n_blocks <= 0 or k_treatments <= 1:
        raise ValueError("n_blocks > 0 and k_treatments > 1 are required.")
    return float(chi2 / (n_blocks * (k_treatments - 1)))
