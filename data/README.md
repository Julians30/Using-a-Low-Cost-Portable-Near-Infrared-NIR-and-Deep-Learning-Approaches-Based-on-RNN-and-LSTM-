# Data

The analytical dataset used by this study is publicly available from **Mendeley Data, Version 2**:

- Ramírez-Morales, I. (2019), *NIR spectra of poultry eggs at different storage days ranging from 0 to 21.*
- DOI: `10.17632/6hn67h2trb.2`
- License stated by Mendeley Data: **CC BY 4.0**.

The raw spectral CSV is intentionally **not duplicated in this repository**. Readers should obtain it from the authoritative Mendeley Data record.

Expected analytical file structure:

- 660 rows
- 30 unique eggs (`sample`)
- 22 storage days (`storage_days`, 0–21)
- 331 spectral columns (`Spectra_740` … `Spectra_1070`)
- SCiO NIR spectral range: 740–1070 nm

Frozen analytical SHA-256 anchor:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

The hash defines the exact computational input used by the frozen pipeline. A local copy with a different hash must not be treated as the manuscript dataset without investigation.

## Source-data licensing

Mendeley Data Version 2 states a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license for the source dataset. This source-data license is distinct from any software license selected for repository code or from licensing of derived documentation.

See [`docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md) and [`docs/SUBMISSION_DATA_LICENSE_NOTE.md`](../docs/SUBMISSION_DATA_LICENSE_NOTE.md).

## Frozen split metadata

The repository includes the exact egg-disjoint outer/inner split assignments and split manifest in `data/frozen_splits/`. These files contain partition metadata rather than spectral measurements and are necessary to reproduce the manuscript exactly.

Frozen split-manifest SHA-256:

`fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717`

See `docs/DATA_DICTIONARY.md`, `docs/REPRODUCIBILITY.md`, and `docs/QUICKSTART.md` before running the analysis.
