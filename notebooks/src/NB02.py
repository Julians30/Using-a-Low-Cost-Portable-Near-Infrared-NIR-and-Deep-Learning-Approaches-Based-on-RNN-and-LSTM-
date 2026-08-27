"""NB02 — Frozen egg-disjoint group splits.

Public source-only export from the frozen analysis notebook used for the NIR shell-egg manuscript.
Notebook outputs, execution counts, Colab runtime metadata, and packaging-only cells are intentionally excluded.
Execute this source in a persistent Python namespace through the matching notebook wrapper.
"""

# ---- Original notebook code cell 1 ----
from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')
RAW_DIR = PROJECT_ROOT / '01_DATA_RAW'
SPLIT_DIR = PROJECT_ROOT / '03_SPLITS_FROZEN'

DATA_FILE = RAW_DIR / 'dataset_egg_storage_RAW.csv'
OUTER_FILE = SPLIT_DIR / 'outer_group_assignment_seed2026.csv'
AUDIT_FILE = SPLIT_DIR / 'outer_split_audit.csv'
MANIFEST_FILE = SPLIT_DIR / 'split_manifest.json'

required = [DATA_FILE, OUTER_FILE, AUDIT_FILE, MANIFEST_FILE]
for p in required:
    assert p.exists(), f'Missing required file: {p}'

df = pd.read_csv(DATA_FILE)
outer = pd.read_csv(OUTER_FILE)
audit_saved = pd.read_csv(AUDIT_FILE)
manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))

print('Dataset:', df.shape)
print('Outer assignment:', outer.shape)
print(json.dumps({k: manifest[k] for k in [
    'outer_folds','outer_seed','inner_folds','inner_seed_rule'
]}, indent=2))

# ---- Original notebook code cell 3 ----
# Audit 1 — each egg assigned to exactly one outer test fold
assert outer['sample'].is_unique, 'An egg appears more than once in the outer assignment.'
assert set(outer['sample']) == set(df['sample'].unique()), 'Outer assignment does not cover exactly all eggs.'
assert sorted(outer['outer_fold'].unique()) == [1,2,3,4,5]
assert outer.groupby('outer_fold')['sample'].nunique().eq(6).all()

display(outer.sort_values(['outer_fold','sample']))
print('PASS: 30 eggs assigned once; 6 test eggs per outer fold.')

# ---- Original notebook code cell 4 ----
# Audit 2 — reconstruct train/test membership and verify zero leakage
audit_rows = []

for outer_fold in range(1, 6):
    test_eggs = set(outer.loc[outer['outer_fold'] == outer_fold, 'sample'])
    train_eggs = set(df['sample'].unique()) - test_eggs

    overlap = train_eggs & test_eggs

    train_df = df[df['sample'].isin(train_eggs)]
    test_df = df[df['sample'].isin(test_eggs)]

    audit_rows.append({
        'outer_fold': outer_fold,
        'n_train_eggs': len(train_eggs),
        'n_test_eggs': len(test_eggs),
        'n_train_spectra': len(train_df),
        'n_test_spectra': len(test_df),
        'train_unique_days': train_df['storage_days'].nunique(),
        'test_unique_days': test_df['storage_days'].nunique(),
        'group_overlap_count': len(overlap)
    })

    assert len(overlap) == 0
    assert len(train_eggs) == 24
    assert len(test_eggs) == 6
    assert len(train_df) == 24 * 22
    assert len(test_df) == 6 * 22
    assert train_df['storage_days'].nunique() == 22
    assert test_df['storage_days'].nunique() == 22

audit_now = pd.DataFrame(audit_rows)
display(audit_now)
print('PASS: zero egg-level leakage in all outer folds.')

# ---- Original notebook code cell 5 ----
# Audit 3 — validate inner assignments for each outer fold
inner_audit_rows = []

for outer_fold in range(1, 6):
    inner_file = SPLIT_DIR / f'inner_group_assignment_outer{outer_fold:02d}.csv'
    assert inner_file.exists(), f'Missing inner file: {inner_file}'
    inner = pd.read_csv(inner_file)

    outer_test_eggs = set(outer.loc[outer['outer_fold'] == outer_fold, 'sample'])
    outer_train_eggs = set(df['sample'].unique()) - outer_test_eggs

    assert inner['sample'].is_unique
    assert set(inner['sample']) == outer_train_eggs
    assert sorted(inner['inner_fold'].unique()) == [1,2,3,4]
    assert inner.groupby('inner_fold')['sample'].nunique().eq(6).all()
    assert set(inner['sample']).isdisjoint(outer_test_eggs)

    for inner_fold in range(1, 5):
        val_eggs = set(inner.loc[inner['inner_fold'] == inner_fold, 'sample'])
        train_eggs = outer_train_eggs - val_eggs
        overlap = train_eggs & val_eggs

        inner_audit_rows.append({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'n_inner_train_eggs': len(train_eggs),
            'n_inner_val_eggs': len(val_eggs),
            'n_inner_train_spectra': int(df['sample'].isin(train_eggs).sum()),
            'n_inner_val_spectra': int(df['sample'].isin(val_eggs).sum()),
            'group_overlap_count': len(overlap),
            'outer_test_egg_intrusion': len(val_eggs & outer_test_eggs)
        })

        assert len(overlap) == 0
        assert len(train_eggs) == 18
        assert len(val_eggs) == 6
        assert len(val_eggs & outer_test_eggs) == 0

inner_audit = pd.DataFrame(inner_audit_rows)
display(inner_audit)
print('PASS: all inner folds are leakage-safe and nested inside outer training data.')

# ---- Original notebook code cell 6 ----
# Audit 4 — verify frozen split-file hashes against manifest
def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

hash_rows = []
for file_name, expected_hash in manifest['files'].items():
    path = SPLIT_DIR / file_name
    assert path.exists(), f'Frozen file missing: {path}'
    observed_hash = sha256_file(path)
    hash_rows.append({
        'file': file_name,
        'expected_sha256': expected_hash,
        'observed_sha256': observed_hash,
        'match': expected_hash == observed_hash
    })

hash_audit = pd.DataFrame(hash_rows)
display(hash_audit[['file','match']])
assert hash_audit['match'].all(), 'At least one frozen split file has changed.'
print('PASS: frozen split files match their manifest hashes.')

# ---- Original notebook code cell 7 ----
# Save reproducibility audit without altering the frozen assignment files
audit_out = SPLIT_DIR / 'NB02_runtime_split_audit.csv'
audit_now.to_csv(audit_out, index=False)

inner_audit_out = SPLIT_DIR / 'NB02_runtime_inner_split_audit.csv'
inner_audit.to_csv(inner_audit_out, index=False)

hash_audit_out = SPLIT_DIR / 'NB02_runtime_hash_audit.csv'
hash_audit.to_csv(hash_audit_out, index=False)

print('Saved runtime audits:')
print(audit_out)
print(inner_audit_out)
print(hash_audit_out)
