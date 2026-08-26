# Notebooks

The reproducible pipeline is executed in the following frozen order:

1. `NB01_DATA_AUDIT.ipynb`
2. `NB02_FROZEN_GROUP_SPLITS.ipynb`
3. `NB03_CHEMOMETRIC_BASELINES.ipynb`
4. `NB04_DEEP_LEARNING_BENCHMARK.ipynb`
5. `NB05_WAVELENGTH_ORDER_ABLATION.ipynb`
6. `NB06_STATISTICAL_ROBUSTNESS.ipynb`
7. `NB07_PRACTICAL_APPLICABILITY.ipynb`
8. `NB08_PUBLICATION_FIGURES_TABLES.ipynb`

The notebooks will be added as cleaned source notebooks with outputs removed where appropriate to keep the repository compact. Standardized frozen result files are retained separately so that reported values can be audited without relying on stored notebook cell outputs.

Do not regenerate the outer/inner partitions independently. Use the frozen files in `data/frozen_splits/`.

See `docs/ANALYSIS_PROTOCOL.md` and `docs/REPRODUCIBILITY.md` for the execution contract.
