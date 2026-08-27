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

## Public notebook format

The eight public `.ipynb` files are lightweight execution notebooks. The scientific source for each stage is stored under `notebooks/src/`. Larger stages are split into ordered source fragments (`part01`, `part02`, ...); the wrapper executes those fragments sequentially in the same Python namespace. NB01 and NB02 are additionally provided as consolidated source files.

Stored notebook cell outputs, execution counts, user-specific Colab runtime metadata, and submission-package/export-only cells are excluded from the public execution wrappers and source fragments. The scientific analysis code, frozen validation logic, model-selection boundaries, statistical procedures, and publication-output generation are retained. Standardized frozen result files are stored separately so reported values can be audited without relying on stored notebook outputs.

**Do not regenerate the outer/inner partitions independently.** Use the frozen files in `data/frozen_splits/`.

See `docs/ANALYSIS_PROTOCOL.md`, `docs/REPRODUCIBILITY.md`, and `docs/QUICKSTART.md` for the execution contract and data layout.
