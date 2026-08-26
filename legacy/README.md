# Legacy Analysis

The repository originally contained a single notebook, `EggSpectrum_Analysis.ipynb`, based on the earlier analytical workflow.

That notebook is retained for **provenance and transparency**, but it is not the source of the manuscript's current performance claims.

## Why it is legacy

The earlier workflow used a random row-level split. Because the dataset contains repeated measurements of the same 30 eggs over 22 storage days, row-level splitting can place spectra from the same egg in both training and test sets. This violates the biological independence assumption required for the primary generalization claim.

The reconstructed manuscript therefore replaces the earlier evaluation with:

- frozen egg-disjoint outer folds
- egg-disjoint inner model selection
- training-only preprocessing
- OOF predictions
- multi-seed deep-learning evaluation
- egg-level inference
- wavelength-order ablation
- reproducibility hashes and standardized output packages

The legacy notebook should be treated as historical context only.

A later repository-cleanup commit will relocate the original root notebook into this directory without rewriting its contents, preserving its exact Git blob/history as far as possible.
