# Publication Tables

These CSV files are frozen exports from NB08 and are intended as the numerical source for the editable manuscript tables.

| Manuscript table | Repository file | Purpose |
|---|---|---|
| Table 1 | `Table_1_Study_Design.csv` | Dataset, SCiO acquisition, and leakage-safe validation design |
| Table 2 | `Table_2_Modeling_Framework.csv` | Model families and analytical roles |
| Table 3 | `Table_3_OOF_Performance.csv` | OOF MAE, RMSE, R², bias, median AE, and operational tolerances |
| Table 4 | `Table_4_Statistical_Robustness.csv` | Egg-level MAE, bootstrap intervals, and average ranks |
| Table 4 post-hoc block | `Table_4B_Top3_PostHoc.csv` | Paired comparisons among SVR, PLSR, and ANN; integrated into Table 4 in the manuscript |
| Table 5 | `Table_5_Wavelength_Order_Ablation.csv` | Original/reversed/shuffled wavelength-order results |
| Table 6 | `Table_6_Practical_Applicability.csv` | Accuracy, complexity, size, CPU latency, and throughput |

## Statistical interpretation guardrail

SVR has the lowest point-estimate OOF MAE, but the paired egg-level NB06 analysis did **not** detect significant differences among SVR, PLSR, and ANN after Holm correction. Therefore these tables must not be used to claim statistical superiority of SVR over PLSR or ANN.

## Formatting

The CSV files deliberately retain full numerical precision. The MDPI/Foods Word manuscript may display rounded values, but the underlying frozen values remain available here for audit and reproducibility.
