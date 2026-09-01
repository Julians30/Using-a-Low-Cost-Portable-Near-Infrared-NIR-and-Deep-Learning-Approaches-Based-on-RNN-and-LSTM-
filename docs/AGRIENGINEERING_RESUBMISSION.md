# AgriEngineering Resubmission Note

**Journal:** AgriEngineering (MDPI)  
**Article type:** Article  
**Status date:** 1 September 2026  
**Previous AgriEngineering manuscript ID:** `agriengineering-4541576`  
**New manuscript ID:** pending assignment in SuSy

## Manuscript title

**Determination of the Storage Time of Shell Eggs at Ambient Temperature Using a Low-Cost Portable Near-Infrared (NIR) Spectrometer and Deep Learning Approaches Based on Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Architectures**

This repository is the public reproducibility package linked to the substantially revised manuscript prepared for resubmission to **AgriEngineering** after the Editor-in-Chief invited the authors to resubmit a substantially revised version.

## What is scientifically new in the revised manuscript

The revised work is a **secondary computational reanalysis** of the previously published and openly available SCiO NIR dataset. No new eggs or spectra were acquired. The scientific contribution is methodological and computational.

The principal revisions are:

1. **Leakage-safe nested egg-disjoint validation.** The egg is treated as the independent biological unit, and repeated measurements from the same egg do not cross training/test partitions.
2. **Training-only model selection.** Preprocessing, hyperparameter selection, and deep-learning epoch selection are restricted to outer-training eggs.
3. **Expanded benchmark.** PLSR, SVR, ANN, SimpleRNN, LSTM, and BiLSTM are compared under identical frozen outer folds.
4. **Egg-level statistical inference.** Friedman testing, paired Wilcoxon tests with Holm correction, effect-size reporting, and whole-egg bootstrap uncertainty are used without treating repeated spectra as independent replicates.
5. **Row-level leakage diagnostic.** A deliberate row-level partitioning analysis quantifies the apparent optimism caused by biological-unit overlap.
6. **Wavelength-order ablation.** Original, reversed, and fixed target-independent shuffled wavelength orders test whether recurrent architectures benefit from the physical 740–1070 nm sequence.
7. **SVR sensitivity analysis.** A wider post hoc grid tests whether primary boundary selections materially alter the frozen result; it does not replace the primary estimate.
8. **Chronological storage-age phase analysis.** Valid egg-disjoint continuous predictions are mapped to Early, Middle, and Late age bands without relabeling them as freshness, safety, or acceptability classes.
9. **Prediction attenuation analysis.** Predicted-on-observed slopes quantify regression toward the center of the experimental range.
10. **Computational applicability.** Model size, complexity, CPU latency, and throughput are reported as engineering reference metrics.
11. **Conservative interpretation.** SVR has the lowest point-estimate error, but SVR, PLSR, and ANN are treated as a statistically competitive group rather than declaring a unique winner without support.
12. **Explicit limitations.** The manuscript states that external validation across batches, farms, instruments, breeds, seasons, temperatures, and humidity conditions has not been performed.

## Main frozen predictive results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| SVR | **2.195** | **2.716** | **0.817** |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

The primary result is the **egg-disjoint** estimate. The row-level diagnostic is intentionally leaky and is reported only to quantify the effect of biological-unit overlap.

## Repository evidence supporting the manuscript

- `data/frozen_splits/` — frozen nested egg-disjoint assignments and manifests
- `notebooks/` — reviewer-facing NB01–NB08 execution wrappers
- `notebooks/src/` — frozen scientific source
- `protocol/` — frozen protocol and reproducibility snapshot
- `results/statistical_robustness/` — egg-level inferential outputs
- `results/practical_applicability/` — computational applicability outputs
- `results/complementary/` — row-level diagnostic, SVR sensitivity, storage-age, and attenuation analyses
- `tables/publication/` — numerical publication tables
- `figures/publication/FIGURE_CAPTIONS.md` — final figure-caption mapping
- `docs/MANUSCRIPT_REPOSITORY_MAPPING.md` — manuscript-to-repository evidence map

## Data provenance

The original spectral dataset is openly available from **Mendeley Data, Version 2**:

- DOI: `10.17632/6hn67h2trb.2`
- Source-data license: CC BY 4.0

The raw CSV is intentionally not duplicated in this repository. The exact analytical input is anchored with SHA-256:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

## Editorial metadata rule

The repository URL is cited in the AgriEngineering submission manuscript. **Do not rename the repository during peer review.** When SuSy assigns the new AgriEngineering manuscript ID, only editorial metadata should be updated; the frozen scientific source and primary numerical results must not be redefined.
