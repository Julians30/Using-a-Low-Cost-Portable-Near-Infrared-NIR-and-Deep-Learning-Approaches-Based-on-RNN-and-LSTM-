# Publication Tables

These CSV files are the numerical sources for the editable manuscript tables. The final manuscript numbering changed after complementary analyses were added; legacy frozen filenames are retained where useful for provenance and automated regression tests.

## Final manuscript numbering

| Manuscript table | Repository file | Purpose |
|---|---|---|
| Table 1 | `Table_1_Study_Design.csv` | Dataset, SCiO acquisition, and leakage-safe validation design |
| Table 2 | `Table_2_Modeling_Framework.csv` | Model families and analytical roles |
| Table 3 | `Table_3_OOF_Performance.csv` | Pooled egg-disjoint OOF MAE, RMSE, R², bias, median AE, and operational tolerances |
| Table 4 | `Table_4_Row_Level_Diagnostic.csv` | Diagnostic comparison of valid egg-disjoint OOF performance with random row-level evaluation |
| Table 5 | `Table_5_Statistical_Robustness.csv` | Egg-level MAE, bootstrap intervals, and average ranks |
| Table 5 post-hoc block | `Table_5B_Top3_PostHoc.csv` | Paired comparisons among SVR, PLSR, and ANN |
| Table 6 | `Table_6_Wavelength_Order_Ablation.csv` | Original/reversed/shuffled wavelength-order results |
| Table 7 | `Table_7_Storage_Age_Phases.csv` | Chronological Early/Middle/Late storage-age phase classification |
| Table 8 | `Table_8_Practical_Applicability.csv` | Predictive performance, complexity, size, CPU latency, and throughput |

## Legacy frozen aliases retained for regression tests

The pre-revision repository package used the following names before Table 4 and Table 7 were introduced:

- `Table_4_Statistical_Robustness.csv` → same core values now exposed as final `Table_5_Statistical_Robustness.csv`
- `Table_4B_Top3_PostHoc.csv` → same core values now exposed as final `Table_5B_Top3_PostHoc.csv`
- `Table_5_Wavelength_Order_Ablation.csv` → same core values now exposed as final `Table_6_Wavelength_Order_Ablation.csv`
- `Table_6_Practical_Applicability.csv` → same core values now exposed as final `Table_8_Practical_Applicability.csv`

The legacy aliases are intentionally not deleted because automated tests and frozen provenance records refer to them. They are not conflicting scientific results.

## Statistical interpretation guardrails

SVR has the lowest point-estimate OOF MAE, but the paired egg-level NB06 analysis did **not** detect significant differences among SVR, PLSR, and ANN after Holm correction. Therefore these tables must not be used to claim statistical superiority or equivalence among the top three models.

The row-level Table 4 is diagnostic only: all 30 eggs overlapped between training and test in every row-level fold. Table 7 describes chronological storage age, not freshness, safety, acceptance, or rejection.

## Formatting

The CSV files deliberately retain full numerical precision where available. The Foods/MDPI Word manuscript displays rounded values for readability, while the repository retains audit-level numerical sources.
