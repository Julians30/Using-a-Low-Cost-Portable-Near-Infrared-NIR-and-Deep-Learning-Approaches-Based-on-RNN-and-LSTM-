# Raw-Data Provenance and Redistribution Status

## Frozen analytical dataset

The reconstructed analysis uses one exact local CSV identified by SHA-256:

```text
cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e
```

Expected structure:

- 660 rows
- 30 shell eggs (`sample`)
- 22 repeated storage days per egg (`storage_days`, 0–21)
- 331 spectral variables (`Spectra_740` through `Spectra_1070`)
- portable miniaturized SCiO NIR acquisition over 740–1070 nm

This hash, rather than a file name alone, defines the computational input used for the frozen analyses.

## Authoritative public source

The manuscript cites the following public dataset as the source of the analyzed spectra:

Ramírez-Morales, I. (2019). *NIR spectra of poultry eggs at different storage days ranging from 0 to 21.* Mendeley Data, Version 2. DOI: `10.17632/6hn67h2trb.2`.

The Mendeley Data record describes 660 spectral curves from 30 intact brown shell eggs monitored over 22 days, acquired with a handheld SCiO NIR spectrometer from 740 to 1070 nm at approximately 1 nm resolution. These characteristics match the analytical dataset documented in the manuscript and repository.

The same acquisition campaign is scientifically described in Coronel-Reyes et al. (2018), *Determination of egg storage time at room temperature using a low-cost NIR spectrometer and machine learning techniques*, **Computers and Electronics in Agriculture**, 145, 1–10, DOI: `10.1016/j.compag.2017.12.030`.

## License and repository policy

Mendeley Data Version 2 states that the source dataset is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Although the source-data reuse terms are now documented, the raw spectral CSV is intentionally **not duplicated in this repository**. Readers should obtain the data from the authoritative Mendeley Data record and verify the local analytical copy against the frozen SHA-256 above.

The source-data CC BY 4.0 license must not be conflated with the license for repository software or other derived assets. A repository software license can be selected separately before archival release.

## What is redistributed here

The repository redistributes analytical and derived materials needed for audit and reproducibility, including:

- frozen outer egg assignments;
- frozen inner egg assignments;
- split audit tables and SHA-256 manifest;
- model/statistical protocol metadata;
- derived aggregate publication tables;
- compact complementary result summaries;
- reproducibility tests and software utilities.

## Submission-ready provenance fields

- **Scientific experiment reference:** Coronel-Reyes et al. (2018), DOI `10.1016/j.compag.2017.12.030` — VERIFIED
- **Public dataset source:** Mendeley Data Version 2 — VERIFIED
- **Dataset DOI:** `10.17632/6hn67h2trb.2` — VERIFIED
- **Dataset license:** CC BY 4.0 — VERIFIED
- **Repository policy:** raw CSV not duplicated; readers obtain it from Mendeley Data and verify the frozen local SHA-256 — DOCUMENTED
