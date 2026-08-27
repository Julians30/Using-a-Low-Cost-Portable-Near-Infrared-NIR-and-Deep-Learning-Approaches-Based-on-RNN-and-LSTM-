# Manuscript–Repository Evidence Map

This document identifies the computational evidence supporting the major manuscript components. It is intended to make peer-review audit easier and to prevent accidental use of legacy results.

## Frozen primary workflow

| Manuscript component | Primary stage | Repository evidence |
|---|---|---|
| Dataset structure and repeated-measures audit | NB01 | `notebooks/NB01_DATA_AUDIT.ipynb`, `notebooks/src/NB01.py`, `scripts/01_data_audit.py`, `docs/DATA_DICTIONARY.md` |
| Egg-disjoint nested validation | NB02 | `notebooks/NB02_FROZEN_GROUP_SPLITS.ipynb`, `data/frozen_splits/`, `src/nir_eggs/splits.py`, `scripts/02_verify_frozen_splits.py` |
| DummyMean / PLSR / SVR benchmark | NB03 | `notebooks/NB03_CHEMOMETRIC_BASELINES.ipynb`, `notebooks/src/NB03_part*.py`, `src/nir_eggs/`, `tables/publication/Table_3_OOF_Performance.csv` |
| ANN / SimpleRNN / LSTM / BiLSTM benchmark | NB04 | `notebooks/NB04_DEEP_LEARNING_BENCHMARK.ipynb`, `notebooks/src/NB04_part*.py`, `protocol/frozen_protocol.json`, `tables/publication/Table_3_OOF_Performance.csv` |
| Wavelength-order mechanism test | NB05 | `notebooks/NB05_WAVELENGTH_ORDER_ABLATION.ipynb`, `notebooks/src/NB05_part*.py`, `src/nir_eggs/ablation.py`, `tables/publication/Table_6_Wavelength_Order_Ablation.csv` |
| Egg-level inferential comparison | NB06 | `notebooks/NB06_STATISTICAL_ROBUSTNESS.ipynb`, `notebooks/src/NB06_part*.py`, `src/nir_eggs/statistics.py`, `results/statistical_robustness/`, `tables/publication/Table_5_Statistical_Robustness.csv` |
| Practical tolerance / complexity / latency | NB07 | `notebooks/NB07_PRACTICAL_APPLICABILITY.ipynb`, `notebooks/src/NB07_part*.py`, `results/practical_applicability/`, `tables/publication/Table_8_Practical_Applicability.csv` |
| Frozen primary publication tables/figure specifications | NB08 | `notebooks/NB08_PUBLICATION_FIGURES_TABLES.ipynb`, `notebooks/src/NB08_part*.py`, `tables/publication/`, `figures/publication/FIGURE_CAPTIONS.md` |

The public NB01–NB08 wrappers checkout the immutable core-source commit `bcdf89dc8f3ad3ca17068210bb8c733748e5a653` before execution. The corresponding cleaned scientific-source SHA-256 anchors are recorded in `notebooks/notebook_source_manifest.json`.

## Complementary manuscript analyses

These analyses were added only after the primary NB01–NB08 workflow and primary OOF predictions had been frozen. They are supporting sensitivity/applied analyses and do **not** replace the primary egg-disjoint estimates.

| Manuscript component | Role | Public repository evidence |
|---|---|---|
| Random row-level partition diagnostic | Quantifies biological-unit overlap and apparent optimism | `results/complementary/row_level_overlap_audit.csv`, `results/complementary/row_level_vs_eggdisjoint.csv`, `tables/publication/Table_4_Row_Level_Diagnostic.csv` |
| Wider-grid SVR analysis | Tests whether NB03 boundary selections materially alter generalization | `results/complementary/svr_sensitivity_summary.csv`, `results/complementary/svr_sensitivity_selected_configs.csv` |
| Chronological storage-age phase analysis | Applied interpretation of valid egg-disjoint continuous OOF predictions | `results/complementary/storage_phase_classification_summary.csv`, `results/complementary/storage_phase_per_class_metrics.csv`, `tables/publication/Table_7_Storage_Age_Phases.csv` |
| Predicted-on-observed attenuation | Quantifies regression toward the center of the experimental range | `results/complementary/predicted_on_observed_attenuation_slopes.csv` |
| Final revised figure captions | Aligns Figures 1–6 with final manuscript interpretation | `figures/publication/FIGURE_CAPTIONS.md` |

The complementary numerical outputs required to audit the manuscript claims are public in this repository. The author-executed complementary/revision notebooks are retained in the project computational records; they were **not** included in the submitted Supplementary Material ZIP and are therefore not represented here as submitted supplementary files. This distinction avoids conflating post-freeze sensitivity analyses with the frozen NB01–NB08 primary workflow.

## Critical interpretation rules

1. **660 spectra are not 660 independent biological samples.** The independent unit is the egg (`sample`), n = 30.
2. **Main performance estimates are OOF and egg-disjoint.** Models fitted to all data may be used only for engineering descriptors, never as generalization estimates.
3. **SVR has the lowest point-estimate MAE, but is not statistically superior to PLSR or ANN under the frozen NB06 analysis.**
4. **Non-significance is not treated as proof of equivalence.** No confirmatory equivalence margin was prespecified.
5. **The row-level result is diagnostic, not a valid alternative validation design.** Its complete egg overlap evaluates new rows from familiar eggs rather than genuinely unseen eggs.
6. **Wider-grid SVR results are sensitivity evidence only.** They do not replace the frozen NB03 result.
7. **Early/Middle/Late are chronological storage-age phases.** They must not be described as freshness, safety, acceptance, or rejection classes.
8. **Wavelength-order ablation is mechanistic.** The fixed shuffled order is target-independent; preprocessing is performed in physical wavelength order before reordering.
9. **CPU latency is environment-specific.** It is not a direct SCiO, mobile-phone, or embedded-device latency claim.
10. **External validation was not performed.** The scope is unseen eggs from the same acquisition campaign.
11. **The legacy notebook is provenance only.** Current manuscript values must trace to the reconstructed pipeline and the final tables/results.

## Notebook source provenance

`notebooks/notebook_source_manifest.json` separates immutable scientific-source integrity anchors from reviewer-facing wrapper files. Wrapper documentation may improve during review, but the frozen core source and primary numerical results are not redefined.
