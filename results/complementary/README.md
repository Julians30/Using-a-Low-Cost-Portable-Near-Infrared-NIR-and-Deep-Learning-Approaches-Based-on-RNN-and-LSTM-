# Complementary Analyses

This directory contains compact numerical outputs for analyses added **after** the NB01–NB08 primary workflow had been frozen. They are reported in the manuscript as complementary/post hoc sensitivity and applied analyses and **do not replace the primary egg-disjoint estimates**.

## Included analyses

- `row_level_overlap_audit.csv` — verifies that every random row-level fold placed all 30 eggs in both training and test partitions.
- `row_level_vs_eggdisjoint.csv` — compares fixed-configuration row-level performance with the frozen egg-disjoint OOF estimates.
- `svr_sensitivity_summary.csv` — compares the frozen primary SVR with a wider-grid sensitivity analysis under the same egg-disjoint nested splits.
- `svr_sensitivity_selected_configs.csv` — fold-specific hyperparameters selected by the wider-grid sensitivity analysis.
- `storage_phase_classification_summary.csv` — accuracy, balanced accuracy, macro-F1, weighted-F1, and Cohen's kappa for chronological storage-age phases.
- `storage_phase_per_class_metrics.csv` — one-vs-rest sensitivity, specificity, precision, and F1 for Early (0–7 d), Middle (8–14 d), and Late (15–21 d).
- `predicted_on_observed_attenuation_slopes.csv` — predicted-on-observed slopes and whole-egg bootstrap 95% confidence intervals for SVR, PLSR, and ANN.

## Interpretation guardrails

1. The row-level analysis is a **diagnostic of biological-unit overlap**, not a valid alternative estimate of generalization.
2. The wider SVR grid is a **sensitivity analysis** and does not replace the frozen NB03 SVR result.
3. Early/Middle/Late are **chronological storage-age bands**, not freshness, safety, acceptability, or rejection classes.
4. Attenuation slopes quantify compression toward the center of the experimental response range; they do not by themselves identify a physicochemical mechanism.
5. The study scope remains generalization to unseen eggs from the **same acquisition campaign**; no external transfer across farms, instruments, batches, breeds, seasons, temperatures, or humidity regimes was established.

The source notebooks used to generate these complementary outputs are supplied with the manuscript submission package. The public repository keeps the frozen core workflow and the compact audit outputs needed to verify the reported numerical claims.
