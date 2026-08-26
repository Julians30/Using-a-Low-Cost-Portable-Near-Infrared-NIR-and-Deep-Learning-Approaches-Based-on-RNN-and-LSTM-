# Reproducibility Guide

## Scope

This document defines the execution contract for reproducing the computational analysis associated with the shell-egg storage-time manuscript. The goal is not merely to re-run code, but to preserve the same biological unit of analysis, data partitions, model-selection boundaries and statistical interpretation used in the manuscript.

## Core rule

**The egg is the independent biological unit.**

The dataset contains 660 spectra, but these are repeated observations from 30 eggs measured over 22 storage days. Any validation that randomly splits rows can place repeated measurements from the same egg in both training and test sets. The manuscript therefore uses frozen egg-disjoint partitions throughout.

## Expected data layout

The raw CSV must contain:

- `storage_days`
- `sample`
- 331 spectral columns named from `Spectra_740` through `Spectra_1070`

Expected dimensions: 660 rows × 333 columns.

Expected biological balance:

- 30 unique eggs
- 22 unique storage days
- one spectrum per egg-day combination

## Frozen hashes

The manuscript reconstruction uses the following integrity anchors:

- raw dataset SHA-256: `cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`
- frozen split-manifest SHA-256: `fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717`

A reproduction should fail loudly if these hashes do not match the frozen release inputs.

## Execution order

Run the notebooks in this exact order:

1. `NB01_DATA_AUDIT.ipynb`
2. `NB02_FROZEN_GROUP_SPLITS.ipynb`
3. `NB03_CHEMOMETRIC_BASELINES.ipynb`
4. `NB04_DEEP_LEARNING_BENCHMARK.ipynb`
5. `NB05_WAVELENGTH_ORDER_ABLATION.ipynb`
6. `NB06_STATISTICAL_ROBUSTNESS.ipynb`
7. `NB07_PRACTICAL_APPLICABILITY.ipynb`
8. `NB08_PUBLICATION_FIGURES_TABLES.ipynb`

Do not skip ahead by regenerating inputs independently. Later notebooks consume frozen outputs produced by earlier stages.

## Validation design

### Outer evaluation

Five egg-disjoint outer folds are frozen a priori. Each fold contains:

- 24 training eggs / 528 spectra
- 6 unseen test eggs / 132 spectra

The same outer assignment is used for every model.

### Inner model selection

Within each outer-training set, four egg-disjoint inner folds are used for preprocessing and hyperparameter/epoch selection. Each inner split contains:

- 18 inner-training eggs / 396 spectra
- 6 inner-validation eggs / 132 spectra

Outer-test eggs are never used for preprocessing, hyperparameter selection, epoch selection or early stopping.

## Preprocessing candidates

The frozen candidate set is:

- Raw
- SNV
- MSC
- Savitzky–Golay smoothing
- Savitzky–Golay first derivative

Any fitted reference or scaling parameter must be derived from the corresponding training partition only. In particular:

- MSC reference spectrum: training-only
- `StandardScaler`: training-only

## Models

### Chemometrics / machine learning

- DummyMean
- PLSR
- RBF-SVR

### Deep learning

- ANN
- SimpleRNN
- LSTM
- BiLSTM

Deep-learning final seeds are `2026`, `2027`, and `2028`.

## Deep-learning interpretation

For recurrent networks, the 331×1 input axis is the ordered **wavelength axis**, not a temporal trajectory. The repeated storage days are separate target-labeled observations and are never presented as one temporal sequence to the recurrent model.

## Wavelength-order ablation

NB05 compares:

1. original wavelength order (740→1070 nm)
2. reversed order
3. one fixed, target-independent shuffled order

Critical implementation rule: spectral preprocessing is performed in the true physical wavelength order **before** reordering the processed features. No re-tuning is performed for reversed or shuffled conditions.

## Performance metrics

Primary metric: MAE in days.

Secondary metrics:

- RMSE
- pooled OOF R²
- bias
- median absolute error
- percentage within ±1 day
- percentage within ±2 days
- percentage within ±3 days

MAPE is not used as a primary metric because storage day 0 makes percentage errors ill-defined.

## Statistical inference

The inferential unit is the egg, not the spectrum and not the random seed.

For deterministic models, MAE is calculated for each egg across its 22 OOF predictions.

For deep-learning models, MAE is calculated separately for each seed within each egg, then the three seed-specific egg MAEs are averaged. This prevents seeds from being treated as biological replicates and avoids giving a seed-averaged prediction ensemble an inferential advantage.

Primary confirmatory analysis:

- Friedman test across six substantive models
- Kendall's W
- 15 paired Wilcoxon signed-rank comparisons
- Holm correction across the 15 comparisons
- paired egg-level bootstrap, 10,000 resamples

A non-significant pairwise difference is described as **not statistically significant**, not as proof of equivalence.

## Practical applicability

NB07 uses frozen OOF predictions for all predictive-performance summaries. Full-data PLSR/SVR refits are permitted only for engineering descriptors such as model size and CPU latency; they must never be reported as generalization performance.

Operational clipping to `[0, 21]` is secondary sensitivity analysis only. Primary performance metrics remain unclipped.

## Publication outputs

NB08 generates publication figures and tables without retraining or selecting models. Figure captions are external to image files. The final figure package uses PNG/TIFF high-resolution outputs.

## Reproduction checklist

A valid reproduction should confirm all of the following:

- 30 unique eggs
- 22 storage days
- 660 spectra
- 331 wavelength variables
- zero egg overlap between train/test folds
- identical outer-fold mapping across NB03/NB04/NB05
- 1,980 NB03 OOF rows
- 7,920 NB04 seed-wise OOF rows
- 23,760 NB05 all-order seed-wise OOF rows
- completed NB06 statistical package
- completed NB07 practical package
- publication-only NB08 with no model retraining

## Reproducibility boundary

This workflow evaluates generalization to unseen eggs **within the same acquisition campaign**. It is not external validation across farms, instruments, breeds, seasons, batches or environmental regimes. Such validation is required before deployment claims.
