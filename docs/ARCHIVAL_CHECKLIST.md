# Archival Release Checklist

This checklist defines the minimum conditions before creating a citable GitHub/Zenodo-style release for the manuscript companion repository.

## Scientific freeze

- [x] NB01 data structure audited.
- [x] NB02 outer and inner egg-disjoint partitions frozen.
- [x] NB03 chemometric/machine-learning benchmark frozen.
- [x] NB04 deep-learning benchmark frozen.
- [x] NB05 wavelength-order ablation frozen.
- [x] NB06 egg-level statistical analysis frozen.
- [x] NB07 practical-applicability analysis frozen.
- [x] NB08 publication tables/figure specifications frozen.
- [x] Main manuscript values cross-checked against frozen publication tables.

## Reproducibility controls

- [x] Dataset SHA-256 documented.
- [x] Split-manifest SHA-256 documented and automatically verified.
- [x] Per-file split hashes automatically verified.
- [x] Reusable source package installable via `pyproject.toml`.
- [x] Lightweight CI tests pass after normal package installation.
- [x] CLI command verifies frozen nested splits.
- [x] CLI command audits an authorized local copy of the raw dataset.
- [x] Legacy row-level analysis preserved separately for provenance.
- [x] Manuscript-to-repository evidence map documented.

## Still required before archival release

- [ ] Verify authoritative raw-dataset source and redistribution terms.
- [ ] Decide whether the raw spectral CSV can legally be redistributed.
- [ ] Add cleaned NB01–NB08 notebook files to the repository or document an archival source from which their recorded SHA-256 copies can be obtained.
- [ ] Decide final repository name.
- [ ] Select a software-code license explicitly; do not apply it automatically to third-party/source data.
- [ ] Decide whether publication PNG previews are stored in Git; keep large TIFF submission masters outside normal Git history unless necessary.
- [ ] Freeze final manuscript version and ensure all displayed values match `tables/publication/`.
- [ ] Update `CITATION.cff` with final repository/release metadata.
- [ ] Create a tagged release only after all above items are resolved.
- [ ] If using Zenodo or another archive, add DOI after the archived release exists.
- [ ] Update manuscript Data Availability / Code Availability statements with final persistent links.

## Reviewer-facing acceptance test

From a clean clone with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test]'
nir-eggs-verify-splits
pytest
```

Expected outcome: all tests and hash checks pass without modifying frozen files.

## Release principle

Do not create a nominally "reproducible" release if the repository contains code whose provenance is unclear, unpublished raw data with unresolved redistribution rights, or numerical tables that do not exactly match the submitted manuscript.
