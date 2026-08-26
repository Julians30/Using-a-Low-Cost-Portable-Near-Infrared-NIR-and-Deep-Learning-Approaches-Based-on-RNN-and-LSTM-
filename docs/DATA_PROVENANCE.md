# Raw-Data Provenance and Redistribution Status

## Frozen analytical dataset

The reconstructed analysis uses one exact CSV file identified by SHA-256:

```text
cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e
```

Expected structure:

- 660 rows
- 30 shell eggs (`sample`)
- 22 repeated storage days per egg (`storage_days`, 0–21)
- 331 spectral variables (`Spectra_740` through `Spectra_1070`)
- portable miniaturized SCiO NIR acquisition

This hash, rather than a file name alone, defines the computational input used for the frozen analyses.

## Redistribution status

The raw spectral CSV is **not currently redistributed in this repository**. Before archival release, the authors must document the authoritative data source and confirm that its license/terms permit republication of the raw spectra.

Until that verification is complete:

1. do not commit the raw CSV to this repository;
2. do not assign a new license to the raw spectral measurements;
3. do not imply that the repository license, once selected, automatically covers the source dataset;
4. keep the dataset SHA-256 in all reproducibility documentation so an authorized local copy can be verified bit-for-bit.

## What is safely redistributed here

The repository does redistribute analytical metadata that do not contain the spectral measurements themselves, including:

- frozen outer egg assignments;
- frozen inner egg assignments;
- split audit tables;
- split SHA-256 manifest;
- model/statistical protocol metadata;
- derived aggregate publication tables and compact result summaries.

## Required archival action

Before the public archival release or final Data Availability Statement is frozen, complete the following fields:

- **Authoritative dataset source:** TO BE VERIFIED
- **Persistent public URL / DOI:** TO BE VERIFIED
- **Original dataset license or reuse terms:** TO BE VERIFIED
- **Whether raw-file redistribution is permitted:** TO BE VERIFIED
- **If redistribution is not permitted, exact reader access procedure:** TO BE VERIFIED

The manuscript should not state that the raw data are openly redistributed through GitHub until these items are resolved.
