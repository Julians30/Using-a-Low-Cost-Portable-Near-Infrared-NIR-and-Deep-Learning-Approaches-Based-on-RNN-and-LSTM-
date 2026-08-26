# Reproducibility Quickstart

This quickstart is designed for a reviewer or reader who wants to verify the computational contract before attempting the expensive deep-learning stages.

## 1. Clone the repository

```bash
git clone https://github.com/Julians30/Using-a-Low-Cost-Portable-Near-Infrared-NIR-and-Deep-Learning-Approaches-Based-on-RNN-and-LSTM-.git
cd Using-a-Low-Cost-Portable-Near-Infrared-NIR-and-Deep-Learning-Approaches-Based-on-RNN-and-LSTM-
```

## 2. Create a Python 3.11 environment

For a lightweight audit:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[test]'
```

For the complete deep-learning/core-notebook environment:

```bash
pip install -e '.[full]'
```

The repository also includes `environment.yml` for Conda users.

## 3. Verify the frozen nested partitions

```bash
nir-eggs-verify-splits
```

A valid result confirms:

- exact split-manifest SHA-256
- all per-file split hashes
- 5 outer folds with 24 train / 6 unseen test eggs
- 4 inner folds with 18 train / 6 validation eggs
- zero egg overlap
- every egg appears in exactly one outer test fold

## 4. Audit the raw dataset

Obtain the study dataset from Mendeley Data (Version 2, DOI `10.17632/6hn67h2trb.2`), place the exact CSV at a local path, and run:

```bash
nir-eggs-audit-data --data /path/to/dataset_egg_storage_RAW.csv
```

The command refuses to proceed if the dataset SHA-256 differs from:

```text
cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e
```

A valid dataset has 660 rows, 30 eggs, 22 days, and 331 wavelength variables from 740 to 1070 nm.

## 5. Run the lightweight automated tests

```bash
pytest
```

These tests verify the core metrics/statistics, train-fitted preprocessing behavior, frozen model grids, frozen network-capacity constants, wavelength-order ablation logic, split integrity, and selected publication-table values.

## 6. Frozen primary analysis order

The primary scientific execution order is:

```text
NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08
```

The deep-learning stages are computationally expensive and were originally executed in Google Colab Pro. Reproducing the primary manuscript claims requires preserving the frozen partitions, seeds and training-only selection boundaries documented in `protocol/frozen_protocol.json` and `docs/ANALYSIS_PROTOCOL.md`.

## 7. Verify final manuscript values without retraining

The numerical sources for the final manuscript are under:

```text
tables/publication/
results/statistical_robustness/
results/practical_applicability/
results/complementary/
```

The final manuscript numbering is documented in `tables/publication/README.md` and `docs/MANUSCRIPT_REPOSITORY_MAPPING.md`.

Key complementary checks that can be verified directly from CSV files are:

- all 30 eggs overlap between training and test in every random row-level fold;
- row-level MAE is 5.1–7.2% lower for SVR/PLSR/ANN than under valid egg-disjoint evaluation;
- the wider-grid SVR sensitivity analysis does not improve the frozen primary SVR estimate;
- SVR chronological storage-age phase accuracy is 78.6%, with macro-F1 0.788 and Cohen's κ 0.680;
- predicted-on-observed slopes for SVR, PLSR and ANN are below 1 with whole-egg bootstrap intervals below 1.

The complementary source notebooks are supplied with the manuscript submission package; their public CSV outputs are kept separate from the frozen NB01–NB08 core so that sensitivity analyses are not mistaken for primary model selection.

## Reproducibility boundary

The study evaluates unseen eggs within one acquisition campaign. It does not constitute external validation across farms, instruments, batches, breeds, seasons, temperatures, humidity regimes, or other acquisition domains. Early/Middle/Late outputs are chronological storage-age phases and must not be interpreted as independently validated freshness or safety classes.
