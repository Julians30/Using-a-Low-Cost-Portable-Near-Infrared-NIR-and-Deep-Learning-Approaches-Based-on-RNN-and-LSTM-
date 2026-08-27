"""NB01 — Data audit.

Public source-only export from the frozen analysis notebook used for the NIR shell-egg manuscript.
Notebook outputs, execution counts, Colab runtime metadata, and packaging-only cells are intentionally excluded.
Execute this source in a persistent Python namespace through the matching notebook wrapper.
"""

# ---- Original notebook code cell 1 ----
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----
# 2. Imports and frozen paths
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')
RAW_DIR = PROJECT_ROOT / '01_DATA_RAW'
PROCESSED_DIR = PROJECT_ROOT / '02_DATA_PROCESSED'
FIG_DIR = PROJECT_ROOT / '06_FIGURES'

AUDIT_DIR = PROCESSED_DIR / 'NB01_AUDIT'
AUDIT_FIG_DIR = FIG_DIR / 'NB01_AUDIT'

for p in [AUDIT_DIR, AUDIT_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATA_FILE = RAW_DIR / 'dataset_egg_storage_RAW.csv'

assert PROJECT_ROOT.exists(), f'Project root not found: {PROJECT_ROOT}'
assert DATA_FILE.exists(), f'Raw dataset not found: {DATA_FILE}'

print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_FILE:', DATA_FILE)

# ---- Original notebook code cell 3 ----
# 3. File fingerprint — SHA-256
def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

sha256 = sha256_file(DATA_FILE)
file_size = DATA_FILE.stat().st_size

print('SHA-256:', sha256)
print('File size (bytes):', file_size)

(AUDIT_DIR / 'dataset_fingerprint.txt').write_text(
    f'file={DATA_FILE.name}\nsha256={sha256}\nsize_bytes={file_size}\n',
    encoding='utf-8'
)

# ---- Original notebook code cell 4 ----
# 4. Load dataset without altering it
df = pd.read_csv(DATA_FILE)

print('Shape:', df.shape)
display(df.head())

# ---- Original notebook code cell 5 ----
# 5. Identify target, group and spectral variables
TARGET = 'storage_days'
GROUP = 'sample'

assert TARGET in df.columns, f'Missing target column: {TARGET}'
assert GROUP in df.columns, f'Missing group column: {GROUP}'

spectral_cols = [c for c in df.columns if c.startswith('Spectra_')]
wavelengths = np.array([int(c.split('_', 1)[1]) for c in spectral_cols])

print('Rows:', len(df))
print('Columns:', df.shape[1])
print('Eggs:', df[GROUP].nunique())
print('Storage days:', df[TARGET].nunique())
print('Spectral variables:', len(spectral_cols))
print('Wavelength min/max:', wavelengths.min(), wavelengths.max())

# ---- Original notebook code cell 6 ----
# 6. Structural assertions expected from the published dataset
expected = {
    'rows': 660,
    'eggs': 30,
    'days': 22,
    'spectral_variables': 331,
    'wavelength_min': 740,
    'wavelength_max': 1070,
}

observed = {
    'rows': int(len(df)),
    'eggs': int(df[GROUP].nunique()),
    'days': int(df[TARGET].nunique()),
    'spectral_variables': int(len(spectral_cols)),
    'wavelength_min': int(wavelengths.min()),
    'wavelength_max': int(wavelengths.max()),
}

comparison = pd.DataFrame({
    'expected': pd.Series(expected),
    'observed': pd.Series(observed)
})
comparison['match'] = comparison['expected'] == comparison['observed']
display(comparison)

assert comparison['match'].all(), 'Dataset does not match the frozen expected structure.'

# ---- Original notebook code cell 7 ----
# 7. Wavelength-axis audit
wl_diff = np.diff(wavelengths)

wavelength_audit = {
    'strictly_increasing': bool(np.all(wl_diff > 0)),
    'all_1nm_steps': bool(np.all(wl_diff == 1)),
    'unique_wavelengths': int(len(np.unique(wavelengths))),
    'duplicate_wavelength_count': int(len(wavelengths) - len(np.unique(wavelengths))),
}

print(json.dumps(wavelength_audit, indent=2))
assert wavelength_audit['strictly_increasing']
assert wavelength_audit['all_1nm_steps']
assert wavelength_audit['duplicate_wavelength_count'] == 0

# ---- Original notebook code cell 8 ----
# 8. Missing, infinite and numeric-type audit
missing_by_col = df.isna().sum()
missing_total = int(missing_by_col.sum())

spectral_numeric = df[spectral_cols].apply(pd.to_numeric, errors='coerce')
coercion_missing = int(spectral_numeric.isna().sum().sum())

spectral_array = spectral_numeric.to_numpy(dtype=float)
inf_total = int(np.isinf(spectral_array).sum())

print('Missing values in full table:', missing_total)
print('Non-numeric / coerced-to-NaN spectral cells:', coercion_missing)
print('Infinite spectral values:', inf_total)

missing_by_col[missing_by_col > 0].to_csv(AUDIT_DIR / 'missing_values_by_column.csv')

# ---- Original notebook code cell 9 ----
# 9. Repeated-measurement structure by egg
egg_summary = (
    df.groupby(GROUP)[TARGET]
      .agg(n_rows='size', day_min='min', day_max='max', n_unique_days='nunique')
      .reset_index()
)

display(egg_summary)

egg_summary.to_csv(AUDIT_DIR / 'egg_structure.csv', index=False)

print('Unique n_rows values:', sorted(egg_summary['n_rows'].unique()))
print('Unique n_unique_days values:', sorted(egg_summary['n_unique_days'].unique()))

# ---- Original notebook code cell 10 ----
# 10. Egg × day completeness
egg_day_counts = (
    df.groupby([GROUP, TARGET])
      .size()
      .rename('n_rows')
      .reset_index()
)

duplicate_egg_day = egg_day_counts.query('n_rows != 1').copy()

complete_grid = (
    pd.MultiIndex.from_product(
        [sorted(df[GROUP].unique()), sorted(df[TARGET].unique())],
        names=[GROUP, TARGET]
    )
    .to_frame(index=False)
    .merge(egg_day_counts, on=[GROUP, TARGET], how='left')
)

complete_grid['n_rows'] = complete_grid['n_rows'].fillna(0).astype(int)
missing_egg_day = complete_grid.query('n_rows == 0').copy()

print('Egg-day combinations expected:', 30 * 22)
print('Observed unique egg-day combinations:', len(egg_day_counts))
print('Missing egg-day combinations:', len(missing_egg_day))
print('Egg-day combinations with count != 1:', len(duplicate_egg_day))

complete_grid.to_csv(AUDIT_DIR / 'egg_day_completeness.csv', index=False)
duplicate_egg_day.to_csv(AUDIT_DIR / 'egg_day_nonunique.csv', index=False)
missing_egg_day.to_csv(AUDIT_DIR / 'egg_day_missing.csv', index=False)

assert len(missing_egg_day) == 0, 'Missing egg-day combinations detected.'
assert len(duplicate_egg_day) == 0, 'Duplicate or repeated egg-day rows detected.'

# ---- Original notebook code cell 11 ----
# 11. Exact duplicate-row and duplicate-spectrum audit
n_exact_duplicate_rows = int(df.duplicated().sum())

# Spectral duplicates regardless of egg/day metadata
spectral_hash = pd.util.hash_pandas_object(df[spectral_cols], index=False)
duplicate_spectra_mask = spectral_hash.duplicated(keep=False)
duplicate_spectra = df.loc[duplicate_spectra_mask, [GROUP, TARGET]].copy()
duplicate_spectra['spectral_hash'] = spectral_hash[duplicate_spectra_mask].astype(str).values

print('Exact duplicated full rows:', n_exact_duplicate_rows)
print('Rows involved in exact duplicated spectra:', len(duplicate_spectra))

duplicate_spectra.to_csv(AUDIT_DIR / 'duplicate_spectra_candidates.csv', index=False)

# ---- Original notebook code cell 12 ----
# 12. Spectral descriptive audit
spectral_desc = pd.DataFrame({
    'wavelength_nm': wavelengths,
    'mean': spectral_numeric.mean(axis=0).values,
    'std': spectral_numeric.std(axis=0, ddof=1).values,
    'min': spectral_numeric.min(axis=0).values,
    'max': spectral_numeric.max(axis=0).values,
    'range': (spectral_numeric.max(axis=0) - spectral_numeric.min(axis=0)).values,
})

spectral_desc['is_constant'] = spectral_desc['std'].fillna(0).eq(0)

display(spectral_desc.head())
print('Constant spectral variables:', int(spectral_desc['is_constant'].sum()))

spectral_desc.to_csv(AUDIT_DIR / 'spectral_descriptive_by_wavelength.csv', index=False)

# ---- Original notebook code cell 13 ----
# 13. Target distribution
target_counts = (
    df[TARGET]
      .value_counts()
      .sort_index()
      .rename_axis(TARGET)
      .reset_index(name='n_spectra')
)

display(target_counts)
target_counts.to_csv(AUDIT_DIR / 'storage_day_counts.csv', index=False)

assert target_counts['n_spectra'].nunique() == 1, 'Storage days are not equally represented.'

# ---- Original notebook code cell 14 ----
# 14. Figure — all individual spectra
fig, ax = plt.subplots(figsize=(10, 6))
X = spectral_numeric.to_numpy(dtype=float)

for i in range(X.shape[0]):
    ax.plot(wavelengths, X[i], alpha=0.12, linewidth=0.7)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('All individual NIR spectra — raw dataset')
fig.tight_layout()

out = AUDIT_FIG_DIR / 'NB01_all_individual_spectra.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print('Saved:', out)

# ---- Original notebook code cell 15 ----
# 15. Figure — mean spectrum by storage day
day_mean = df.groupby(TARGET)[spectral_cols].mean()

fig, ax = plt.subplots(figsize=(10, 6))
for day, row in day_mean.iterrows():
    ax.plot(wavelengths, row.to_numpy(dtype=float), alpha=0.75, linewidth=1.0, label=str(day))

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean reflectance')
ax.set_title('Mean NIR spectrum by storage day')
# Avoid an oversized legend in the main panel
handles, labels = ax.get_legend_handles_labels()
if len(labels) <= 12:
    ax.legend(title='Day', ncol=2, fontsize=8)
fig.tight_layout()

out = AUDIT_FIG_DIR / 'NB01_mean_spectrum_by_storage_day.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print('Saved:', out)

# ---- Original notebook code cell 16 ----
# 16. Figure — grand mean ± 1 SD
grand_mean = spectral_numeric.mean(axis=0).to_numpy(dtype=float)
grand_sd = spectral_numeric.std(axis=0, ddof=1).to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths, grand_mean, linewidth=1.5, label='Mean')
ax.fill_between(wavelengths, grand_mean - grand_sd, grand_mean + grand_sd, alpha=0.2, label='±1 SD')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Grand mean spectrum and between-observation variability')
ax.legend()
fig.tight_layout()

out = AUDIT_FIG_DIR / 'NB01_grand_mean_sd.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print('Saved:', out)

# ---- Original notebook code cell 17 ----
# 17. Save frozen audit summary
summary = {
    'file_name': DATA_FILE.name,
    'sha256': sha256,
    'file_size_bytes': int(file_size),
    'shape_rows': int(df.shape[0]),
    'shape_columns': int(df.shape[1]),
    'group_column': GROUP,
    'target_column': TARGET,
    'n_eggs': int(df[GROUP].nunique()),
    'n_storage_days': int(df[TARGET].nunique()),
    'storage_day_min': int(df[TARGET].min()),
    'storage_day_max': int(df[TARGET].max()),
    'n_spectral_variables': int(len(spectral_cols)),
    'wavelength_min_nm': int(wavelengths.min()),
    'wavelength_max_nm': int(wavelengths.max()),
    'wavelength_step_all_1nm': bool(np.all(np.diff(wavelengths) == 1)),
    'missing_total': missing_total,
    'spectral_infinite_total': inf_total,
    'exact_duplicate_rows': n_exact_duplicate_rows,
    'duplicate_spectra_rows': int(len(duplicate_spectra)),
    'missing_egg_day_combinations': int(len(missing_egg_day)),
    'nonunique_egg_day_combinations': int(len(duplicate_egg_day)),
    'constant_spectral_variables': int(spectral_desc['is_constant'].sum()),
    'python_version': sys.version,
    'platform': platform.platform(),
    'pandas_version': pd.__version__,
    'numpy_version': np.__version__,
}

with open(AUDIT_DIR / 'dataset_audit_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

display(pd.Series(summary, name='value').to_frame())
print('\nNB01 completed successfully.')
print('Audit outputs:', AUDIT_DIR)
print('Audit figures:', AUDIT_FIG_DIR)
