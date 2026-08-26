# Leakage-Safe NIR Egg Storage-Time Prediction

Reproducibility repository for the manuscript:

**Determination of the Storage Time of Shell Eggs at Ambient Temperature Using a Low-Cost Portable Near-Infrared (NIR) Spectrometer and Deep-Learning Approaches Based on Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Architectures**

> Status: manuscript reconstruction and reproducibility package prepared for submission to **Foods (MDPI)**.

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

The frozen analytical framework compares:

- DummyMean
- Partial Least Squares Regression (PLSR)
- Support Vector Regression (SVR)
- Artificial Neural Network (ANN)
- SimpleRNN
- LSTM
- BiLSTM

The recurrent architectures operate along the ordered wavelength axis; this axis is spectral, not temporal.

## Main frozen OOF results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| SVR | **2.195** | **2.716** | **0.817** |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

These values are descriptive pooled out-of-fold point estimates. Egg-level inference showed that **SVR, PLSR and ANN did not differ significantly from one another after Holm correction**, whereas all three recurrent architectures had significantly larger errors than the competitive top group.

## Why this repository is different from the original analysis

The initial notebook used a random row-level train/test split. Because each egg contributed repeated spectra across storage days, the same biological unit could occur in both training and test sets. The reconstructed workflow therefore uses frozen egg-disjoint outer and inner partitions, training-only preprocessing/tuning, out-of-fold predictions, multi-seed deep-learning evaluation, egg-level statistical inference and wavelength-order ablation.

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

Each stage is designed to consume frozen outputs from the preceding stages. Performance is never re-estimated from a model fitted to all 30 eggs.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── results/
│   └── README.md
├── figures/
│   └── README.md
├── tables/
│   └── README.md
├── docs/
│   ├── ANALYSIS_PROTOCOL.md
│   ├── DATA_DICTIONARY.md
│   └── REPRODUCIBILITY.md
└── legacy/
    └── README.md
```

The frozen notebooks, splits and selected result files will be added in the corresponding directories as the repository rebuild is completed.

## Reproducibility principles

1. **Independent biological unit = egg.** The 660 spectra are repeated observations, not 660 independent samples.
2. **Frozen outer folds.** The same five egg-disjoint test folds are used for every model.
3. **Nested model selection.** Preprocessing, hyperparameters and epoch budgets are selected using only outer-training eggs.
4. **OOF-only performance reporting.** Main predictive metrics are calculated only from predictions for unseen outer-test eggs.
5. **Seeds are algorithmic repeats, not biological replicates.** Deep-learning inference is aggregated at the egg level.
6. **Mechanistic ablation.** Original, reversed and target-independent shuffled wavelength orders are compared without retuning.
7. **Hash-based auditability.** Frozen data/split manifests and standardized result packages are checked using SHA-256.

## Statistical analysis

The primary inferential outcome is per-egg MAE across the 22 storage days. The confirmatory workflow uses:

- Friedman test across the six substantive models
- Kendall's W for global effect size
- paired Wilcoxon signed-rank tests
- Holm correction across pairwise comparisons
- paired egg-level bootstrap with 10,000 resamples

No claim of model equivalence is made from a non-significant difference because no confirmatory equivalence margin was prespecified.

## Wavelength-order ablation

The deep-learning models are evaluated under three spectral-order conditions:

- original: 740 → 1070 nm
- reversed: 1070 → 740 nm
- fixed target-independent shuffled order

Preprocessing is always performed in the true physical wavelength order before any feature reordering. This prevents Savitzky–Golay operations from being applied to an artificial wavelength sequence.

## Practical applicability

The repository also documents operational tolerance (±1, ±2 and ±3 days), performance across early/middle/late storage periods, prediction clipping sensitivity, model complexity and CPU latency. CPU timings are implementation-specific reference measurements and are **not** claims about SCiO, smartphone or embedded-device latency.

## Data

The raw dataset is not duplicated in this repository until its redistribution status/source is explicitly documented. See [`data/README.md`](data/README.md). Frozen split assignments and their manifests can be redistributed independently because they contain identifiers and partition metadata rather than spectral measurements.

## Legacy analysis

The pre-reconstruction notebook is retained for transparency and provenance. It must not be used as the source of the manuscript's current performance claims. See [`legacy/README.md`](legacy/README.md).

## Citation

Please use [`CITATION.cff`](CITATION.cff) when citing this repository. The final DOI/journal metadata will be added after publication or archival release.

## License

A software/data license will be specified before the archival release. Until then, no additional redistribution rights should be inferred beyond those granted by GitHub and the original data source.
