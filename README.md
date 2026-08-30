# Leakage-Safe NIR Egg Storage-Time Prediction

Reproducibility repository for the manuscript:

**Determination of the Storage Time of Shell Eggs at Ambient Temperature Using a Low-Cost Portable Near-Infrared (NIR) Spectrometer and Deep Learning Approaches Based on Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Architectures**

**Authors:** Julián Coronel-Reyes, Alexander Fernando Haro Sarango, Vanessa Vergara-Lozano, Carlota Delgado-Vera, and Héctor Ramiro Carvajal Romero.

> **Submission status (30 August 2026):** manuscript submitted as **Original Research** to **Frontiers in Artificial Intelligence — AI in Food, Agriculture and Water** (Manuscript ID: **1982643**), within the Research Topic **Artificial Intelligence, Sensing, and Robotic Innovations in Animal and Food Systems for Next-Generation Processing and Safety**. The primary computational workflow is scientifically frozen; repository documentation may still receive non-scientific metadata clarification during editorial processing.

## Scientific scope

This repository documents a secondary computational reanalysis of a public portable-NIR shell-egg dataset. The **egg is the independent biological unit**; repeated spectra from the same egg are not treated as independent samples.

- 30 intact brown-shell eggs
- 22 storage days per egg (0–21)
- 660 spectra
- 331 spectral variables
- SCiO NIR range: 740–1070 nm
- 5 frozen outer egg-disjoint folds
- 4 egg-disjoint inner folds for training-only model selection
- deep-learning seeds: 2026, 2027, 2028

The inferential scope is **previously unseen eggs from the same acquisition campaign**. This study is not an external validation across farms, batches, instruments, breeds, seasons, temperatures, humidity regimes, or other acquisition domains.

## Data provenance

The source dataset is publicly available from **Mendeley Data, Version 2**:

- Ramírez-Morales, I. (2019), *NIR spectra of poultry eggs at different storage days ranging from 0 to 21*
- DOI: `10.17632/6hn67h2trb.2`
- Source-data license stated by Mendeley Data: **CC BY 4.0**

The raw spectral CSV is intentionally **not duplicated** here. Readers should obtain it from the authoritative Mendeley record and verify the exact computational input with SHA-256:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

See `data/README.md` and `docs/DATA_PROVENANCE.md`.

## Frozen primary workflow

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

Primary model selection is performed only with outer-training eggs. Main predictive performance is calculated from frozen egg-disjoint outer-fold out-of-fold (OOF) predictions.

The public `.ipynb` files under `notebooks/` are lightweight execution wrappers. To prevent drift, they checkout the immutable core-source commit:

`bcdf89dc8f3ad3ca17068210bb8c733748e5a653`

Frozen scientific source hashes are recorded in `notebooks/notebook_source_manifest.json`.

## Main frozen OOF results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| SVR | **2.195** | **2.716** | **0.817** |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

SVR has the lowest pooled OOF point-estimate error, but egg-level inference does **not** support treating SVR, PLSR, and ANN as statistically distinct winners after Holm correction. Non-significance is not interpreted as equivalence.

## Complementary analyses

After NB01–NB08 and the primary OOF predictions were frozen, compact secondary analyses were added without changing the primary model-selection or inference rules:

- **Row-level partition diagnostic:** every random row-level fold placed all 30 eggs in both training and test; MAE appeared 5.1–7.2% lower for SVR/PLSR/ANN than under valid egg-disjoint evaluation.
- **Wider-grid SVR sensitivity:** did not improve the frozen primary SVR estimate.
- **Chronological storage-age phases:** SVR reached 78.6% accuracy, macro-F1 = 0.788, and Cohen's κ = 0.680 for Early (0–7 d), Middle (8–14 d), and Late (15–21 d).
- **Prediction attenuation:** predicted-on-observed slopes were below 1 for SVR, PLSR, and ANN.

These are supporting analyses, not replacements for the frozen primary workflow. Early/Middle/Late are **chronological storage-age bands**, not independently validated freshness, safety, acceptability, or rejection classes.

Auditable numerical outputs are under `results/complementary/` and are mapped to the manuscript in `docs/MANUSCRIPT_REPOSITORY_MAPPING.md`.

## Quick reviewer audit

With Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[test]'
nir-eggs-verify-splits
pytest
```

For the full deep-learning notebook environment:

```bash
pip install -e '.[full]'
```

The lightweight CI/test suite verifies package installation, split integrity and hashes, frozen protocol constraints, core preprocessing/statistical utilities, deterministic wavelength-order logic, selected publication values, and archival metadata. It intentionally does **not** retrain the expensive TensorFlow models on every GitHub Actions run.

See `docs/QUICKSTART.md` for the reviewer-focused workflow.

## Critical interpretation rules

1. **Independent biological unit = egg (n = 30).** The 660 spectra are repeated observations.
2. **Primary performance = egg-disjoint OOF predictions.**
3. **No outer-test information is used for model selection.**
4. **Deep-learning seeds are algorithmic repeats, not biological replicates.**
5. **Non-significance is not evidence of equivalence.**
6. **Row-level splitting is diagnostic and intentionally leaky at the biological-unit level.**
7. **Wavelength-order ablation is mechanistic, not a new tuning route.**
8. **CPU latency values are environment-specific engineering references, not mobile/SCiO deployment benchmarks.**
9. **External validation has not been performed.**

## Repository map

- `data/frozen_splits/` — exact nested egg-disjoint assignments and hash manifest
- `notebooks/` — public reviewer-facing NB01–NB08 wrappers
- `notebooks/src/` — frozen scientific source code/fragments
- `src/nir_eggs/` — reusable reproducibility utilities
- `protocol/` — frozen protocol and reproducibility snapshot
- `results/` — frozen statistical, practical, and complementary outputs
- `tables/publication/` — manuscript numerical tables
- `figures/publication/` — final figure captions and publication mapping
- `docs/` — protocol, data provenance, quickstart, and manuscript-to-repository map
- `tests/` — automated reproducibility checks
- `legacy/` — earlier row-level analysis retained only for provenance

## Citation and archival status

Use `CITATION.cff` when citing this repository. The repository URL is already cited by the submitted manuscript and therefore should **not be renamed during editorial processing or peer review**.

A tagged archival release/DOI has not yet been created. A software-code license has also not yet been selected by the authors; the CC BY 4.0 statement above applies to the **source dataset** as stated by Mendeley Data and must not be conflated with repository software rights.
