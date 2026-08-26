"""Reusable utilities for the NIR-HUEVOS reproducibility repository.

The frozen manuscript results are produced by NB01–NB08.  These utilities
mirror core preprocessing, metric, and statistical operations so they can be
unit-tested independently of the notebooks.
"""

from .metrics import regression_metrics
from .preprocessing import SpectralPreprocessor
from .statistics import holm_adjust, paired_bootstrap_mean_difference

__all__ = [
    "SpectralPreprocessor",
    "regression_metrics",
    "holm_adjust",
    "paired_bootstrap_mean_difference",
]
