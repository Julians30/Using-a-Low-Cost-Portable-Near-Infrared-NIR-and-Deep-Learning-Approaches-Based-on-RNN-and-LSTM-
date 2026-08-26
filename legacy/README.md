# Legacy Analysis

The repository originally contained a single notebook, `EggSpectrum_Analysis.ipynb`, based on the earlier analytical workflow.

That notebook is now preserved as:

`legacy/EggSpectrum_Analysis_original.ipynb`

It is retained for **provenance and transparency**, but it is not the source of the manuscript's current performance claims.

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

The legacy notebook should be treated as historical context only. It was moved using its existing Git blob rather than rewritten, so the file content itself was preserved during the repository reorganization.
