"""Public source fragment for NB03_CHEMOMETRIC_BASELINES.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----
from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----
# Imports, paths, and frozen protocol
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, platform, shutil, subprocess, sys, time, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')
RAW_DIR = PROJECT_ROOT / '01_DATA_RAW'
SPLIT_DIR = PROJECT_ROOT / '03_SPLITS_FROZEN'
RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB03_CHEMOMETRIC_BASELINES'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB03_CHEMOMETRIC_BASELINES'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'

# Clean only this stage's regenerable outputs to prevent stale-file carryover.
for p in [RESULT_DIR, FIG_DIR]:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
ZIP_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = RAW_DIR / 'dataset_egg_storage_RAW.csv'
OUTER_FILE = SPLIT_DIR / 'outer_group_assignment_seed2026.csv'
SPLIT_MANIFEST_FILE = SPLIT_DIR / 'split_manifest.json'

EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'

PREPROCESSING_CANDIDATES = ['raw', 'snv', 'msc', 'sg_smooth', 'sg_deriv1']
SG_WINDOW = 11
SG_POLYORDER = 2

PLS_COMPONENTS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]

SVR_C = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
SVR_EPSILON = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SVR_GAMMA = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]

PRIMARY_METRIC = 'MAE_days'

print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_FILE:', DATA_FILE)
print('RESULT_DIR:', RESULT_DIR)

SEARCH_REVISION = 'v3_final_broad_grid_after_v2_inner_boundary_audit'
GRID_EXPANSION_STOP_RULE = 'FINAL: no further expansion after v3; remaining boundary hits are reported, not retuned.'
NOTEBOOK_FILENAME = 'NB03_CHEMOMETRIC_BASELINES_v3.ipynb'

# ---- Original notebook code cell 3 ----
# Integrity gate — dataset and frozen split hashes
def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

assert DATA_FILE.exists(), f'Missing raw dataset: {DATA_FILE}'
assert OUTER_FILE.exists(), f'Missing frozen outer split: {OUTER_FILE}'
assert SPLIT_MANIFEST_FILE.exists(), f'Missing split manifest: {SPLIT_MANIFEST_FILE}'

dataset_sha = sha256_file(DATA_FILE)
assert dataset_sha == EXPECTED_DATASET_SHA256, (
    f'Dataset hash changed. Expected {EXPECTED_DATASET_SHA256}, observed {dataset_sha}'
)

split_manifest = json.loads(SPLIT_MANIFEST_FILE.read_text(encoding='utf-8'))
hash_audit = []
for file_name, expected_hash in split_manifest['files'].items():
    p = SPLIT_DIR / file_name
    observed = sha256_file(p)
    hash_audit.append({
        'file': file_name,
        'expected_sha256': expected_hash,
        'observed_sha256': observed,
        'match': observed == expected_hash
    })
hash_audit = pd.DataFrame(hash_audit)
assert hash_audit['match'].all(), 'One or more frozen split files changed after NB02.'

hash_audit.to_csv(RESULT_DIR / 'NB03_input_hash_audit.csv', index=False)

print('PASS — dataset SHA-256:', dataset_sha)
print('PASS — all frozen split hashes match NB02 manifest.')

# ---- Original notebook code cell 4 ----
# Load dataset and reconstruct wavelength-ordered predictor matrix
df = pd.read_csv(DATA_FILE)
outer = pd.read_csv(OUTER_FILE)

required_cols = {'sample', 'storage_days'}
assert required_cols.issubset(df.columns)

spectral_cols = [c for c in df.columns if c.startswith('Spectra_')]
assert len(spectral_cols) == 331, f'Expected 331 spectral variables, got {len(spectral_cols)}'

def wavelength_from_col(c):
    return float(c.replace('Spectra_', ''))

spectral_cols = sorted(spectral_cols, key=wavelength_from_col)
wavelengths = np.array([wavelength_from_col(c) for c in spectral_cols], dtype=float)

X_all = df[spectral_cols].to_numpy(dtype=np.float64)
y_all = df['storage_days'].to_numpy(dtype=np.float64)
groups_all = df['sample'].to_numpy()

assert X_all.shape == (660, 331)
assert df['sample'].nunique() == 30
assert df['storage_days'].nunique() == 22
assert set(outer['sample']) == set(df['sample'].unique())

print('X:', X_all.shape, '| eggs:', df['sample'].nunique(), '| days:', df['storage_days'].nunique())
print('Wavelengths:', wavelengths.min(), 'to', wavelengths.max(), 'nm')

# ---- Original notebook code cell 6 ----
# Leakage-safe preprocessing utilities
class SpectralPreprocessor:
    def __init__(self, name, sg_window=11, sg_polyorder=2):
        self.name = name
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.reference_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.name == 'msc':
            self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.name == 'raw':
            return X.copy()

        if self.name == 'snv':
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, ddof=1, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            return (X - mu) / sd

        if self.name == 'msc':
            if self.reference_ is None:
                raise RuntimeError('MSC must be fitted on training data first.')
            ref = self.reference_
            ref_mean = ref.mean()
            ref_centered = ref - ref_mean
            denom = np.dot(ref_centered, ref_centered)
            out = np.empty_like(X, dtype=np.float64)
            for i, x in enumerate(X):
                x_mean = x.mean()
                b = np.dot(ref_centered, x - x_mean) / denom
                if abs(b) < 1e-12:
                    b = 1.0
                a = x_mean - b * ref_mean
                out[i] = (x - a) / b
            return out

        if self.name == 'sg_smooth':
            return savgol_filter(
                X, window_length=self.sg_window, polyorder=self.sg_polyorder,
                deriv=0, axis=1, mode='interp'
            )

        if self.name == 'sg_deriv1':
            return savgol_filter(
                X, window_length=self.sg_window, polyorder=self.sg_polyorder,
                deriv=1, delta=1.0, axis=1, mode='interp'
            )

        raise ValueError(f'Unknown preprocessing: {self.name}')

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ae = np.abs(err)
    return {
        'MAE_days': float(mean_absolute_error(y_true, y_pred)),
        'RMSE_days': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'bias_days': float(np.mean(err)),
        'median_AE_days': float(np.median(ae)),
        'within_1d_pct': float(100*np.mean(ae <= 1.0)),
        'within_2d_pct': float(100*np.mean(ae <= 2.0)),
        'within_3d_pct': float(100*np.mean(ae <= 3.0)),
    }

# ---- Original notebook code cell 7 ----
# Inner fold loader — uses exactly NB02 frozen assignments
def get_outer_train_test_eggs(outer_fold):
    test_eggs = set(outer.loc[outer['outer_fold'] == outer_fold, 'sample'].tolist())
    train_eggs = set(df['sample'].unique()) - test_eggs
    assert len(train_eggs) == 24 and len(test_eggs) == 6
    assert not (train_eggs & test_eggs)
    return train_eggs, test_eggs

def load_inner_assignment(outer_fold):
    p = SPLIT_DIR / f'inner_group_assignment_outer{outer_fold:02d}.csv'
    inner = pd.read_csv(p)
    outer_train_eggs, outer_test_eggs = get_outer_train_test_eggs(outer_fold)
    assert inner['sample'].is_unique
    assert set(inner['sample']) == outer_train_eggs
    assert not (set(inner['sample']) & outer_test_eggs)
    assert inner.groupby('inner_fold')['sample'].nunique().eq(6).all()
    return inner

print('Frozen split structure revalidated.')
