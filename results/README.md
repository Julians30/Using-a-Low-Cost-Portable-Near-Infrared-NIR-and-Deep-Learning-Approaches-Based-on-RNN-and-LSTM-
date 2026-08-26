# Results

This directory is reserved for compact, frozen outputs required to audit the manuscript without re-running every training job.

Planned subdirectories:

- `chemometrics/` — NB03 OOF predictions and pooled/per-egg metrics
- `deep_learning/` — NB04 seed-wise and seed-mean OOF predictions/metrics
- `wavelength_ablation/` — NB05 all-order predictions and order-effect summaries
- `statistical_robustness/` — NB06 egg-level inference tables
- `practical_applicability/` — NB07 operational metrics and CPU latency summaries

The repository will prioritize CSV/JSON/TXT outputs that support the manuscript's exact numerical claims. Large temporary checkpoints and redundant packaging files are excluded.

## Frozen headline results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| SVR | 2.195 | 2.716 | 0.817 |
| PLSR | 2.267 | 2.864 | 0.796 |
| ANN | 2.289 | 2.974 | 0.780 |
| BiLSTM | 4.558 | 5.586 | 0.225 |
| LSTM | 4.710 | 5.572 | 0.229 |
| SimpleRNN | 4.879 | 5.722 | 0.187 |

These are pooled descriptive OOF values. Confirmatory inference is performed at the egg level; see `docs/ANALYSIS_PROTOCOL.md`.
