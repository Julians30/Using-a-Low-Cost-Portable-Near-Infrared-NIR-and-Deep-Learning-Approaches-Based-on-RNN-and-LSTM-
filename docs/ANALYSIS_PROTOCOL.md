# Frozen Analysis Protocol

## Objective

Estimate shell-egg storage time at ambient temperature from portable SCiO NIR spectra while preventing biological-unit leakage and separating descriptive predictive performance from confirmatory egg-level inference.

## NB01 — Data audit

Purpose:

- verify table dimensions and variable names
- confirm 30 eggs × 22 storage days
- check missing / infinite values
- check duplicated egg-day records
- identify the 331 ordered wavelength columns
- perform descriptive spectral exploration only

No model performance is selected in this stage.

## NB02 — Frozen grouped partitions

Five outer egg-disjoint folds are generated and frozen with seed 2026. Every egg appears in exactly one outer-test fold.

Outer-test eggs by fold:

1. 9, 11, 15, 22, 23, 24
2. 4, 10, 12, 16, 19, 29
3. 1, 5, 8, 17, 18, 30
4. 3, 6, 13, 20, 21, 28
5. 2, 7, 14, 25, 26, 27

Each outer-training set is further partitioned into four egg-disjoint inner folds for training-only selection.

## NB03 — Chemometric baselines

Models:

- DummyMean
- PLSR
- RBF-SVR

Preprocessing candidates:

- Raw
- SNV
- MSC
- Savitzky–Golay smoothing
- Savitzky–Golay first derivative

PLSR candidate latent components:

`[2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]`

SVR candidate grid:

- C: `[1, 10, 100, 1000, 10000, 100000]`
- epsilon: `[0.05, 0.1, 0.25, 0.5, 1, 2]`
- gamma: `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`

The grid was frozen before final outer evaluation. Boundary-seeking SVR selections are reported as a limitation rather than triggering a post-hoc grid expansion.

## NB04 — Deep-learning benchmark

Architectures:

- ANN
- SimpleRNN
- LSTM
- BiLSTM

The architectures are compact and fixed a priori. Inner CV selects only preprocessing and epoch budget.

Frozen training rules:

- maximum 300 epochs in inner selection
- early stopping monitor: validation MAE
- patience: 20
- minimum delta: 0.001
- restore best weights: true
- final outer model is trained on all 24 outer-training eggs for the median best epoch selected across the four inner folds
- no outer-test data are used for early stopping
- final seeds: 2026, 2027, 2028

## NB05 — Wavelength-order ablation

Purpose: test whether preserving physical wavelength order provides useful predictive information to ANN and recurrent architectures.

Conditions:

- original
- reversed
- fixed shuffled order

The shuffled permutation is target-independent and fixed with seed 52026.

Preprocessing and scaling are always applied in physical wavelength order before transformed features are reordered.

Original NB04 predictions are reused exactly. Only reversed and shuffled conditions require new final fits. There is no repeated inner tuning.

## NB06 — Statistical robustness

Primary models:

- SVR
- PLSR
- ANN
- BiLSTM
- LSTM
- SimpleRNN

DummyMean is excluded from the primary Friedman test so that a trivial baseline does not dominate the omnibus ranking.

Primary outcome: per-egg MAE across 22 days.

Deep-learning rule: calculate per-egg MAE separately for each seed and average the three seed-specific MAEs before statistical testing.

Confirmatory tests:

- Friedman omnibus test
- Kendall's W
- 15 paired Wilcoxon signed-rank comparisons
- Holm multiplicity correction across all 15 primary pairwise tests
- paired bootstrap by egg, 10,000 resamples, seed 62026

Secondary analyses include substantive-model comparisons against DummyMean and order-effect tests within each deep architecture.

## NB07 — Practical applicability

No new generalization estimate is created.

Outputs include:

- MAE, RMSE, R², bias, median AE
- percentages within ±1, ±2 and ±3 days
- early (0–7), middle (8–14) and late (15–21) storage summaries
- day-specific error and bias
- out-of-range predictions
- secondary clipping sensitivity to [0, 21]
- model complexity / serialized size
- CPU inference-latency reference measurements

PLSR/SVR may be fitted to all 660 observations only to estimate engineering descriptors such as latency and model size. These full-data fits are never used for manuscript generalization claims.

## NB08 — Publication outputs

Purpose: convert frozen outputs into manuscript-ready figures and tables.

Rules:

- no retraining
- no hyperparameter selection
- figure text in English
- no title embedded at the top of any figure
- external captions
- high-resolution PNG and TIFF exports
- publication tables derived directly from frozen result files

## Primary result interpretation

The point-estimate ranking does not by itself determine statistical superiority.

Frozen pooled OOF values:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| SVR | 2.195 | 2.716 | 0.817 |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

Primary egg-level inference supports a competitive SVR/PLSR/ANN group without significant pairwise differences after Holm correction. The three recurrent architectures show significantly larger errors than that group.

## Claim boundary

This is a proof-of-concept evaluation of unseen eggs under the same acquisition campaign. The protocol does not establish transportability across independent farms, instruments, environmental conditions, breeds, batches, or seasons.
