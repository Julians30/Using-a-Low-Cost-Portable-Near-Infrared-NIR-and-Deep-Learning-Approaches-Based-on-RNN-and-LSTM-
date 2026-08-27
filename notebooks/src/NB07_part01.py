"""Public source fragment for NB07_PRACTICAL_APPLICABILITY.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----

from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----

from pathlib import Path
from datetime import datetime, timezone
import ast as pyast
import hashlib, itertools, json, math, os, platform, shutil, subprocess, sys, tempfile, time, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')

DATA_RAW_DIR = PROJECT_ROOT / '01_DATA_RAW'
NB03_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB03_CHEMOMETRIC_BASELINES'
NB04_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB04_DEEP_LEARNING_BENCHMARK'
NB05_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB05_WAVELENGTH_ORDER_ABLATION'
NB06_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB06_STATISTICAL_ROBUSTNESS'

RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB07_PRACTICAL_APPLICABILITY'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB07_PRACTICAL_APPLICABILITY'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'

for p in [RESULT_DIR, FIG_DIR, ZIP_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW_DIR / 'dataset_egg_storage_RAW.csv'

NB03_OOF = NB03_DIR / 'NB03_oof_predictions.csv'
NB03_SELECTED = NB03_DIR / 'NB03_selected_configurations.csv'
NB03_PROTOCOL = NB03_DIR / 'NB03_protocol.json'
NB03_SUMMARY = NB03_DIR / 'NB03_run_summary.json'

NB04_SEEDWISE = NB04_DIR / 'NB04_oof_predictions_seedwise.csv'
NB04_SELECTED = NB04_DIR / 'NB04_selected_configurations.csv'
NB04_COMPLEXITY = NB04_DIR / 'NB04_model_complexity.csv'
NB04_TRAIN_METRICS = NB04_DIR / 'NB04_outer_fold_seed_metrics.csv'
NB04_PROTOCOL = NB04_DIR / 'NB04_protocol.json'
NB04_SUMMARY = NB04_DIR / 'NB04_run_summary.json'

NB05_PROTOCOL = NB05_DIR / 'NB05_protocol.json'
NB05_SUMMARY = NB05_DIR / 'NB05_run_summary.json'

NB06_DESC = NB06_DIR / 'NB06_unified_descriptive_metrics_MAE_RMSE_R2.csv'
NB06_PAIRWISE = NB06_DIR / 'NB06_pairwise_wilcoxon_holm.csv'
NB06_ORDER = NB06_DIR / 'NB06_order_pairwise_wilcoxon_holm.csv'
NB06_PROTOCOL = NB06_DIR / 'NB06_protocol.json'
NB06_SUMMARY = NB06_DIR / 'NB06_run_summary.json'
NB06_STATUS = NB06_DIR / 'EXECUTION_STATUS.json'

NOTEBOOK_FILENAME = 'NB07_PRACTICAL_APPLICABILITY.ipynb'
RUN_REVISION = 'NB07_v1_frozen_practical_applicability'
PACKAGE_SCHEMA = 'NIR-HUEVOS standardized result package v1.5'

EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'
EXPECTED_SPLIT_MANIFEST_SHA256 = 'fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717'

MODELS_MAIN = ['SVR','PLSR','ANN','BiLSTM','LSTM','SimpleRNN']
DEEP_MODELS = ['ANN','SimpleRNN','LSTM','BiLSTM']
FINAL_SEEDS = [2026,2027,2028]
PHYSICAL_MIN_DAY = 0.0
PHYSICAL_MAX_DAY = 21.0

LATENCY_SINGLE_REPEATS = 200
LATENCY_BATCH_REPEATS = 60
LATENCY_WARMUP = 20
LATENCY_SEED = 72026

print('NB07: no requiere GPU para desempeño OOF; la latencia se medirá en CPU.')

# ---- Original notebook code cell 3 ----

def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

required = [
    RAW_CSV,
    NB03_OOF, NB03_SELECTED, NB03_PROTOCOL, NB03_SUMMARY,
    NB04_SEEDWISE, NB04_SELECTED, NB04_PROTOCOL, NB04_SUMMARY,
    NB06_DESC, NB06_PAIRWISE, NB06_ORDER, NB06_PROTOCOL, NB06_SUMMARY, NB06_STATUS
]
for p in required:
    assert p.exists(), f'Falta archivo requerido: {p}'

p03 = json.loads(NB03_PROTOCOL.read_text(encoding='utf-8'))
p04 = json.loads(NB04_PROTOCOL.read_text(encoding='utf-8'))
p06 = json.loads(NB06_PROTOCOL.read_text(encoding='utf-8'))
s03 = json.loads(NB03_SUMMARY.read_text(encoding='utf-8'))
s04 = json.loads(NB04_SUMMARY.read_text(encoding='utf-8'))
s06 = json.loads(NB06_SUMMARY.read_text(encoding='utf-8'))
e06 = json.loads(NB06_STATUS.read_text(encoding='utf-8'))

assert s03.get('status') == 'COMPLETED'
assert s04.get('status') == 'COMPLETED'
assert s06.get('status') == 'COMPLETED'
assert e06.get('status') == 'COMPLETED'

assert sha256_file(RAW_CSV) == EXPECTED_DATASET_SHA256, 'Hash del dataset RAW no coincide.'

def first_key(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

for label, protocol in [('NB03',p03),('NB04',p04),('NB06',p06)]:
    ds = first_key(protocol, 'dataset_sha256', 'dataset_hash_sha256')
    if ds is not None:
        assert ds == EXPECTED_DATASET_SHA256, f'{label}: hash de dataset no coincide.'
    sh = first_key(protocol, 'frozen_split_manifest_sha256', 'split_manifest_sha256')
    if sh is not None:
        assert sh == EXPECTED_SPLIT_MANIFEST_SHA256, f'{label}: hash de splits no coincide.'

audit = pd.DataFrame([{
    'source_file': str(p.relative_to(PROJECT_ROOT)),
    'size_bytes': p.stat().st_size,
    'sha256': sha256_file(p)
} for p in required])
audit.to_csv(RESULT_DIR / 'NB07_input_hash_audit.csv', index=False)

print('PASS — fuentes NB03/NB04/NB06 y dataset RAW verificados.')
display(audit)

# ---- Original notebook code cell 4 ----

# Consolidación OOF descriptiva.
# PLSR/SVR/DummyMean: predicción determinista NB03.
# DL: promedio de las tres predicciones por semilla, SOLO para descripción operativa.

nb03 = pd.read_csv(NB03_OOF)
nb04 = pd.read_csv(NB04_SEEDWISE)

assert len(nb03) == 1980
assert len(nb04) == 7920

base03 = nb03[['sample','storage_days','outer_fold','model','y_pred']].copy()

deep_mean = (
    nb04
    .groupby(['sample','storage_days','outer_fold','model'], as_index=False)
    .agg(y_pred=('y_pred','mean'))
)

assert len(deep_mean) == 2640
assert not deep_mean.duplicated(['sample','storage_days','model']).any()

oof = pd.concat([base03, deep_mean], ignore_index=True)
assert len(oof) == 4620
assert oof['model'].nunique() == 7
assert oof.groupby('model').size().eq(660).all()
assert oof.groupby('model')['sample'].nunique().eq(30).all()
assert oof.groupby('model')['storage_days'].nunique().eq(22).all()

oof['error_days'] = oof['y_pred'] - oof['storage_days']
oof['abs_error_days'] = np.abs(oof['error_days'])
oof['y_pred_clipped'] = oof['y_pred'].clip(PHYSICAL_MIN_DAY, PHYSICAL_MAX_DAY)

oof.to_csv(RESULT_DIR / 'NB07_oof_operational_predictions.csv', index=False)
print('PASS — 4,620 predicciones OOF consolidadas (7 modelos × 660).')

# ---- Original notebook code cell 5 ----

def metrics_block(g, pred_col='y_pred'):
    y = g['storage_days'].to_numpy(float)
    p = g[pred_col].to_numpy(float)
    err = p - y
    ae = np.abs(err)
    return {
        'n_spectra': int(len(g)),
        'n_eggs': int(g['sample'].nunique()),
        'MAE_days': float(mean_absolute_error(y,p)),
        'RMSE_days': float(np.sqrt(mean_squared_error(y,p))),
        'R2': float(r2_score(y,p)),
        'bias_days': float(np.mean(err)),
        'median_AE_days': float(np.median(ae)),
        'within_1d_pct': float(np.mean(ae <= 1.0)*100),
        'within_2d_pct': float(np.mean(ae <= 2.0)*100),
        'within_3d_pct': float(np.mean(ae <= 3.0)*100),
        'pred_min': float(np.min(p)),
        'pred_max': float(np.max(p)),
        'out_of_range_pct': float(np.mean((p < PHYSICAL_MIN_DAY) | (p > PHYSICAL_MAX_DAY))*100)
    }

summary_rows = []
for model_name, g in oof.groupby('model'):
    row = {'model': model_name}
    row.update(metrics_block(g,'y_pred'))
    summary_rows.append(row)

operational = pd.DataFrame(summary_rows)
order = ['SVR','PLSR','ANN','BiLSTM','LSTM','SimpleRNN','DummyMean']
operational['model'] = pd.Categorical(operational['model'], categories=order, ordered=True)
operational = operational.sort_values('model').reset_index(drop=True)
operational['model'] = operational['model'].astype(str)

operational.to_csv(RESULT_DIR / 'NB07_operational_performance_summary.csv', index=False)
display(operational)

# ---- Original notebook code cell 6 ----

# Validación cruzada contra las métricas ya congeladas en NB06.
nb06_desc = pd.read_csv(NB06_DESC)

check_cols = ['MAE_days','RMSE_days','R2']
check = operational.merge(
    nb06_desc[['model'] + check_cols],
    on='model',
    how='left',
    suffixes=('_NB07','_NB06')
)

for metric in check_cols:
    delta = np.abs(check[f'{metric}_NB07'] - check[f'{metric}_NB06'])
    assert np.nanmax(delta) < 1e-9, f'NB07 no reproduce {metric} de NB06.'

check.to_csv(RESULT_DIR / 'NB07_metric_reproduction_audit.csv', index=False)
print('PASS — MAE, RMSE y R² reproducen exactamente NB06.')

# ---- Original notebook code cell 7 ----

# Fases de almacenamiento: descriptivas, no nueva inferencia.
def storage_phase(day):
    if day <= 7:
        return 'Early_0_7'
    elif day <= 14:
        return 'Middle_8_14'
    return 'Late_15_21'

oof['storage_phase'] = oof['storage_days'].apply(storage_phase)

phase_rows = []
for (model_name, phase), g in oof.groupby(['model','storage_phase']):
    row = {'model': model_name, 'storage_phase': phase}
    row.update(metrics_block(g,'y_pred'))
    phase_rows.append(row)
phase_metrics = pd.DataFrame(phase_rows)

phase_order = ['Early_0_7','Middle_8_14','Late_15_21']
phase_metrics['storage_phase'] = pd.Categorical(
    phase_metrics['storage_phase'], categories=phase_order, ordered=True
)
phase_metrics = phase_metrics.sort_values(['model','storage_phase'])
phase_metrics['storage_phase'] = phase_metrics['storage_phase'].astype(str)
phase_metrics.to_csv(RESULT_DIR / 'NB07_metrics_by_storage_phase.csv', index=False)

