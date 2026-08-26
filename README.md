# Leakage-Safe NIR Egg Storage-Time Prediction

Reproducibility repository for the manuscript:

**Determination of the Storage Time of Shell Eggs at Ambient Temperature Using a Low-Cost Portable Near-Infrared (NIR) Spectrometer and Deep-Learning Approaches Based on Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Architectures**

> Status: leakage-safe computational analysis frozen; manuscript and repository package being finalized for submission to **Foods (MDPI)**.

## Quick verification

With Python 3.11, a reviewer can install the lightweight reproducibility package and validate the frozen partitions without TensorFlow:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[test]'
nir-eggs-verify-splits
pytest
```

For the full deep-learning/notebook environment:

```bash
pip install -e '.[full]'
```

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for the reviewer-focused execution guide.

## Study overview

This repository documents a leakage-safe re-evaluation of shell-egg storage-time prediction from spectra acquired with a portable miniaturized **SCiO NIR spectrometer**. The analysis treats the **egg as the independent biological unit**, rather than the individual spectrum.

- 30 shell eggs
- 22 repeated storage days per egg (days 0–21)
- 660 spectra in total
- 331 spectral variables
- 740–1070 nm spectral range
- 5 frozen outer egg-disjoint folds
- 4 egg-disjoint inner folds for training-only model selection
- Deep-learning seeds: 2026, 2027 and 2028

## Models

The frozen analytical framework compares DummyMean, PLSR, RBF-SVR, ANN, SimpleRNN, LSTM and BiLSTM. The recurrent architectures operate along the ordered wavelength axis; this axis is **spectral, not temporal**.

## Main frozen OOF results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| SVR | **2.195** | **2.716** | **0.817** |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

These values are descriptive pooled out-of-fold point estimates. Egg-level inference showed that **SVR, PLSR and ANN did not differ significantly from one another after Holm correction**, whereas each of those three models significantly outperformed the recurrent architectures in the primary pairwise analysis.

## Why this repository differs from the legacy analysis

The original notebook used a random row-level train/test split. Because every egg contributed repeated spectra across storage days, spectra from the same biological unit could occur in both training and test sets. The reconstructed workflow therefore uses frozen egg-disjoint outer and inner partitions, training-only preprocessing/tuning, out-of-fold predictions, multi-seed deep-learning evaluation, egg-level statistical inference and wavelength-order ablation.

The original notebook is preserved under `legacy/` for provenance; it is not the source of the current manuscript claims.

## Reproducible analysis sequence

```text
NB01_DATA_AUDIT
      ↓
NB02_FROZEN_GROUP_SPLITS
      ↓
NB03_CHEMOMETRIC_BASELINES
      ↓
NB04_DEEP_LEARNING_BENCHMARK
      ↓
NB05_WAVELENGTH_ORDER_ABLATION
      ↓
NB06_STATISTICAL_ROBUSTNESS
      ↓
NB07_PRACTICAL_APPLICABILITY
      ↓
NB08_PUBLICATION_FIGURES_TABLES
```

Each stage consumes frozen outputs from the preceding stages. Performance is never re-estimated from a model fitted to all 30 eggs.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── requirements-ci.txt
├── environment.yml
├── data/
│   ├── README.md
│   └── frozen_splits/
├── notebooks/
│   └── notebook_source_manifest.json
├── scripts/
│   ├── 01_data_audit.py
│   └── 02_verify_frozen_splits.py
├── src/nir_eggs/
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── splits.py
│   ├── models.py
│   ├── chemometric.py
│   ├── ablation.py
│   └── cli.py
├── protocol/
│   └── frozen_protocol.json
├── results/
│   ├── statistical_robustness/
│   └── practical_applicability/
├── figures/publication/
│   └── FIGURE_CAPTIONS.md
├── tables/publication/
│   ├── Table_1_Study_Design.csv
│   ├── Table_2_Modeling_Framework.csv
│   ├── Table_3_OOF_Performance.csv
│   ├── Table_4_Statistical_Robustness.csv
│   ├── Table_4B_Top3_PostHoc.csv
│   ├── Table_5_Wavelength_Order_Ablation.csv
│   └── Table_6_Practical_Applicability.csv
├── docs/
│   ├── ANALYSIS_PROTOCOL.md
│   ├── DATA_DICTIONARY.md
│   ├── MANUSCRIPT_REPOSITORY_MAPPING.md
│   ├── QUICKSTART.md
│   └── REPRODUCIBILITY.md
├── tests/
└── legacy/
```

## Reproducibility principles

1. **Independent biological unit = egg.** The 660 spectra are repeated observations, not 660 independent samples.
2. **Frozen outer folds.** The same five egg-disjoint test folds are used for every model.
3. **Nested model selection.** Preprocessing, hyperparameters and epoch budgets are selected using only outer-training eggs.
4. **OOF-only performance reporting.** Main predictive metrics are calculated only from predictions for unseen outer-test eggs.
5. **Seeds are algorithmic repeats, not biological replicates.** Deep-learning inference is aggregated at the egg level.
6. **Mechanistic ablation.** Original, reversed and fixed target-independent shuffled wavelength orders are compared without retuning.
7. **Hash-based auditability.** Frozen data/split manifests and standardized result packages are checked using SHA-256.

## Frozen splits

`data/frozen_splits/` contains the exact outer and inner group assignments used by every model. The manifest itself has frozen SHA-256:

`fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717`

The repository's GitHub Actions workflow verifies both the manifest and its split-file hashes automatically.

## Source code and notebooks

The reusable installable package under `src/nir_eggs/` mirrors the final preprocessing, metrics, statistical utilities, model capacity, split logic and wavelength-order ablation rules. `notebooks/notebook_source_manifest.json` records the eight cleaned source notebooks and their SHA-256 fingerprints. Cleaned notebook copies are defined as copies with output cells and execution counts removed, while scientific source and markdown are preserved.

## Statistical analysis

The primary inferential outcome is per-egg MAE across the 22 storage days. The confirmatory workflow uses Friedman testing, Kendall's W, paired Wilcoxon signed-rank comparisons, Holm correction and paired egg-level bootstrap with 10,000 resamples.

No claim of model equivalence is made from a non-significant difference because no confirmatory equivalence margin was prespecified.

## Wavelength-order ablation

The deep-learning models are evaluated under original (740→1070 nm), reversed (1070→740 nm) and fixed target-independent shuffled spectral orders. Preprocessing is always performed in the true physical wavelength order **before** reordering. This prevents Savitzky–Golay smoothing/derivatives from being applied to an artificial spectral sequence.

## Publication tables and figures

The frozen numerical source for Tables 1–6 is stored under `tables/publication/`. Full numerical precision is retained for audit; the Word manuscript may display rounded values. Figure captions and the no-internal-title editorial rule are documented under `figures/publication/`.

The publication figures are regenerated from frozen outputs rather than treated as independent evidence files.

## Practical applicability

The repository documents operational tolerance (±1, ±2 and ±3 days), early/middle/late storage performance, clipping sensitivity, model complexity and CPU latency. CPU timings are implementation-specific reference measurements and are **not** claims about SCiO, smartphone or embedded-device latency.

## Data

The raw dataset is not duplicated here until redistribution rights/source are explicitly documented. See [`data/README.md`](data/README.md). Its frozen SHA-256 is:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

Frozen split assignments can be redistributed independently because they contain identifiers and partition metadata rather than spectral measurements.

## Automated checks

GitHub Actions installs the project as a normal Python package and validates:

- package installation and CLI availability
- core regression/statistical utilities
- train-only MSC behavior
- exact frozen split-manifest and split-file hashes
- outer/inner egg-disjointness and fold sizes
- frozen search grids and neural-network capacity constants
- deterministic wavelength-order permutation
- frozen publication-table values and statistical interpretation guards

## Citation

Please use [`CITATION.cff`](CITATION.cff) when citing this repository. Final DOI/journal metadata will be added after archival release or publication.

## License

A software/data license will be selected before archival release. Until then, no additional redistribution rights should be inferred beyond those granted by GitHub and the original data source.
