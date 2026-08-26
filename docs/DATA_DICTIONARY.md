# Data Dictionary

## Raw analytical table

Expected file: `dataset_egg_storage_RAW.csv`

Expected shape: **660 × 333**.

| Field | Type | Meaning |
|---|---|---|
| `storage_days` | integer / numeric | Storage time in days; target variable, ranging from 0 to 21. |
| `sample` | integer / categorical identifier | Egg identifier. This is the grouping variable and independent biological-unit identifier. |
| `Spectra_740` … `Spectra_1070` | numeric | SCiO NIR spectral variables corresponding to wavelengths from approximately 740 to 1070 nm. |

## Biological structure

- 30 unique eggs
- 22 repeated storage days per egg
- 660 egg-day observations
- exactly one analytical spectrum per egg-day record in the reconstructed table

The analytical spectrum represents the average of two readings acquired from the same shell position in the original acquisition protocol.

## Target

`storage_days` is treated as a continuous regression target.

Primary unit: days.

The study does not convert storage time into freshness classes for the main analysis.

## Grouping variable

`sample` must be used for all leakage-safe train/validation/test partitioning. Random row-level splitting is not valid for the primary manuscript analysis because repeated measurements from a single egg are biologically dependent.

## Spectral axis

The 331 wavelength variables form an ordered spectral vector. In ANN the vector is treated as a multivariate feature vector. In SimpleRNN/LSTM/BiLSTM, the same wavelength order is represented as a 331×1 sequence solely to test whether recurrent architectures can exploit ordered spectral structure.

The wavelength axis must not be described as a time sequence.

## Frozen integrity hashes

Raw dataset SHA-256:

`cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e`

Frozen split manifest SHA-256:

`fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717`

## Derived files

Derived repository outputs may contain:

- outer fold assignment
- inner fold assignments
- selected preprocessing / hyperparameters
- OOF predictions
- per-egg metrics
- seed-wise predictions
- wavelength-order ablation predictions
- statistical test summaries
- bootstrap confidence intervals
- practical applicability metrics
- publication tables and figures

Derived result files must never be silently substituted for the raw spectral table when rerunning earlier pipeline stages.
