# Results

This directory contains compact outputs required to audit the manuscript without re-running every training job.

Available subdirectories:

- `statistical_robustness/` — NB06 egg-level inference tables and pairwise results
- `practical_applicability/` — NB07 operational metrics, clipping sensitivity, model complexity, and CPU latency summaries
- `complementary/` — post-freeze diagnostic/sensitivity/applied outputs used in the final manuscript without replacing the primary NB01–NB08 estimates

The repository prioritizes CSV/JSON/TXT outputs that support exact numerical claims. Large temporary checkpoints and redundant packaging files are excluded.

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

## Complementary manuscript results

The final manuscript additionally reports analyses performed only after the primary workflow was frozen:

- random row-level splitting produced complete train/test egg overlap in every fold and modestly optimistic performance for the competitive models;
- a wider SVR grid did not improve the frozen primary SVR result;
- valid continuous OOF predictions were summarized into chronological Early/Middle/Late storage-age phases;
- predicted-on-observed slopes quantified attenuation toward the center of the response range.

See `results/complementary/README.md` for exact interpretation boundaries. These analyses are supporting evidence and must not be used to retroactively redefine primary model selection.
