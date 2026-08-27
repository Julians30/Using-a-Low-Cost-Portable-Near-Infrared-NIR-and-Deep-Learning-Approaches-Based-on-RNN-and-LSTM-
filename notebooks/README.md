# Notebooks

The frozen primary pipeline is executed in the following order:

1. `NB01_DATA_AUDIT.ipynb`
2. `NB02_FROZEN_GROUP_SPLITS.ipynb`
3. `NB03_CHEMOMETRIC_BASELINES.ipynb`
4. `NB04_DEEP_LEARNING_BENCHMARK.ipynb`
5. `NB05_WAVELENGTH_ORDER_ABLATION.ipynb`
6. `NB06_STATISTICAL_ROBUSTNESS.ipynb`
7. `NB07_PRACTICAL_APPLICABILITY.ipynb`
8. `NB08_PUBLICATION_FIGURES_TABLES.ipynb`

## Public notebook format

The eight public `.ipynb` files are lightweight reviewer-facing execution wrappers. The scientific source for each primary stage is stored under `notebooks/src/`. Larger stages are split into ordered source fragments (`part01`, `part02`, ...); each wrapper executes those fragments sequentially in the same Python namespace.

To prevent the reproducibility target from drifting as `main` evolves, every primary wrapper checks out the immutable core-source commit:

`bcdf89dc8f3ad3ca17068210bb8c733748e5a653`

The frozen cleaned-source SHA-256 anchors are recorded in `notebook_source_manifest.json` and cross-checked by the repository tests.

Stored notebook cell outputs, execution counts, user-specific Colab runtime metadata, and packaging/export-only cells are excluded from the public wrappers. The scientific analysis code, frozen validation logic, model-selection boundaries, statistical procedures, and publication-output generation are retained.

## Execution environment

The primary scientific notebooks were originally executed in Google Colab/Google Drive and the frozen source intentionally preserves that execution contract. A reviewer does **not** need to rerun the expensive deep-learning stages to audit the manuscript values: frozen result tables, split assignments, protocol metadata, and automated tests are included in the repository.

For a lightweight platform-independent audit, follow `docs/QUICKSTART.md` and run:

```bash
pip install -e '.[test]'
nir-eggs-verify-splits
pytest
```

**Do not regenerate the outer/inner partitions independently.** Use the exact files in `data/frozen_splits/`.

See `docs/ANALYSIS_PROTOCOL.md`, `docs/REPRODUCIBILITY.md`, and `docs/QUICKSTART.md` for the execution contract and data layout.
