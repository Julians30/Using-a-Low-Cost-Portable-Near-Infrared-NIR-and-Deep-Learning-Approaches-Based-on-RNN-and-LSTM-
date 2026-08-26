# Manuscript–Repository Evidence Map

This document identifies the frozen computational source supporting each major manuscript component. It is intended to make peer-review audit easier and to prevent accidental use of legacy results.

| Manuscript component | Primary computational stage | Repository evidence |
|---|---|---|
| Dataset structure and repeated-measures audit | NB01 | `scripts/01_data_audit.py`, `docs/DATA_DICTIONARY.md` |
| Egg-disjoint nested validation | NB02 | `data/frozen_splits/`, `src/nir_eggs/splits.py`, `scripts/02_verify_frozen_splits.py` |
| DummyMean / PLSR / SVR benchmark | NB03 | `src/nir_eggs/preprocessing.py`, `src/nir_eggs/models.py`, `src/nir_eggs/chemometric.py` |
| ANN / SimpleRNN / LSTM / BiLSTM benchmark | NB04 | `src/nir_eggs/models.py`, `protocol/frozen_protocol.json` |
| Wavelength-order mechanism test | NB05 | `src/nir_eggs/ablation.py`, `tables/publication/Table_5_Wavelength_Order_Ablation.csv` |
| Egg-level inferential comparison | NB06 | `src/nir_eggs/statistics.py`, `results/statistical_robustness/`, `tables/publication/Table_4_Statistical_Robustness.csv` |
| Practical tolerance / complexity / latency | NB07 | `results/practical_applicability/`, `tables/publication/Table_6_Practical_Applicability.csv` |
| Final manuscript figures and tables | NB08 | `tables/publication/`, `figures/publication/FIGURE_CAPTIONS.md` |

## Critical interpretation rules

1. **660 spectra are not 660 independent biological samples.** The independent unit is the egg (`sample`), n = 30.
2. **Main performance estimates are OOF and egg-disjoint.** Models fitted to all data may be used only for engineering/deployment descriptors, never as generalization estimates.
3. **SVR has the lowest point-estimate MAE, but is not statistically superior to PLSR or ANN under the frozen NB06 analysis.**
4. **Non-significance is not treated as proof of equivalence.** No confirmatory equivalence margin was prespecified.
5. **Wavelength-order ablation is mechanistic.** The fixed shuffled order is target-independent; preprocessing is performed in physical wavelength order before reordering.
6. **CPU latency is environment-specific.** It is not a direct SCiO, mobile-phone, or embedded-device latency claim.
7. **The legacy notebook is provenance only.** Current manuscript values must trace to the reconstructed pipeline and frozen tables/results.

## Notebook source provenance

`notebooks/notebook_source_manifest.json` records the SHA-256 fingerprints of the cleaned NB01–NB08 source notebooks used for repository mirroring. This enables a reviewer or author to verify that a notebook copy corresponds to the frozen source version even when outputs are stripped for GitHub readability.
