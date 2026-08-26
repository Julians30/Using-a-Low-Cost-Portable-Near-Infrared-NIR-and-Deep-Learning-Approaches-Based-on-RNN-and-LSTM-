# Data

The raw spectral CSV is intentionally **not redistributed here yet** until its authoritative source and redistribution status are documented explicitly for the archival release.

Expected analytical file:

`dataset_egg_storage_RAW.csv`

Expected SHA-256:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

Expected structure:

- 660 rows
- 30 unique eggs (`sample`)
- 22 storage days (`storage_days`, 0–21)
- 331 spectral columns (`Spectra_740` … `Spectra_1070`)

The hash defines the exact computational input used by the frozen pipeline. A local file with a different hash must not be treated as the manuscript dataset.

## Redistribution guardrail

See [`docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md). Until the authoritative source/license is verified, the raw CSV must not be committed to GitHub and no future repository software license should be assumed to cover those spectral measurements.

## Frozen split metadata

The repository includes the exact egg-disjoint outer/inner split assignments and split manifest in `data/frozen_splits/`. These files contain partition metadata rather than spectral measurements and are necessary to reproduce the manuscript exactly.

Frozen split-manifest SHA-256:

`fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717`

See `docs/DATA_DICTIONARY.md`, `docs/REPRODUCIBILITY.md`, and `docs/QUICKSTART.md` before running the analysis.
