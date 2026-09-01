# Archival Release Checklist

This checklist distinguishes what is already sufficient for peer-review reproducibility from what is still needed only for a final archival release.

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
- [x] Complementary analyses clearly separated from the frozen primary workflow.

## Reproducibility controls completed

- [x] Authoritative public dataset source documented: Mendeley Data Version 2, DOI `10.17632/6hn67h2trb.2`.
- [x] Source-data license documented as CC BY 4.0.
- [x] Raw spectral CSV intentionally not duplicated; exact analytical SHA-256 documented.
- [x] Split-manifest SHA-256 documented and automatically verified.
- [x] Per-file split hashes automatically verified.
- [x] Public NB01–NB08 reviewer-facing notebooks committed.
- [x] Frozen primary scientific source committed under `notebooks/src/`.
- [x] Primary wrappers pinned to immutable core-source commit `bcdf89dc8f3ad3ca17068210bb8c733748e5a653`.
- [x] Reusable source package installable via `pyproject.toml`.
- [x] Lightweight CI tests pass after normal package installation.
- [x] CLI command verifies frozen nested splits.
- [x] CLI command audits an authorized local copy of the raw dataset.
- [x] Legacy row-level analysis preserved separately for provenance.
- [x] Manuscript-to-repository evidence map documented.
- [x] `CITATION.cff` aligned with the manuscript author order and title.
- [x] README, `CITATION.cff`, manuscript evidence map, and journal-specific resubmission note aligned to the **AgriEngineering** resubmission target (1 September 2026; previous manuscript ID `agriengineering-4541576`).

## Still required only before a final archival release

- [ ] Add the new AgriEngineering manuscript ID after SuSy assigns it.
- [ ] Select a software-code license explicitly. Do not apply the source dataset's CC BY 4.0 automatically to repository code.
- [ ] Decide whether to create a tagged release and persistent archival DOI (e.g., Zenodo). This is not required for the current peer-review submission.
- [ ] Add final journal citation/DOI to `CITATION.cff` after publication metadata exists.
- [ ] If editorially permitted after acceptance, update persistent archival links in the final paper; otherwise preserve the already-cited GitHub URL.

## Repository-name rule during peer review

The AgriEngineering resubmission manuscript cites the current GitHub URL. **Do not rename this repository during peer review**, because doing so could create avoidable link ambiguity. GitHub normally redirects renamed repositories, but reproducibility documentation should not depend on that behavior while the manuscript is under review.

## Reviewer-facing acceptance test

From a clean clone with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[test]'
nir-eggs-verify-splits
pytest
```

Expected outcome: all lightweight tests and frozen split/hash checks pass without modifying frozen scientific files.

The CI intentionally does not retrain the computationally expensive TensorFlow models at every commit. Their frozen source, seeds, model-capacity constraints, selection boundaries, and result artifacts remain auditable in the repository.

## Release principle

Do not create a nominally archival release until software licensing and final publication metadata are resolved. Those remaining archival items do not prevent peer review of the current reproducibility package.
