"""Public source fragment for NB04_DEEP_LEARNING_BENCHMARK.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----
from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----
# Imports, fixed protocol, paths, and GPU gate
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, math, os, platform, random, shutil, subprocess, sys, time, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')
RAW_DIR = PROJECT_ROOT / '01_DATA_RAW'
SPLIT_DIR = PROJECT_ROOT / '03_SPLITS_FROZEN'
RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB04_DEEP_LEARNING_BENCHMARK'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB04_DEEP_LEARNING_BENCHMARK'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'
CHECKPOINT_DIR = RESULT_DIR / '_CHECKPOINT'

for p in [RESULT_DIR, FIG_DIR, ZIP_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = RAW_DIR / 'dataset_egg_storage_RAW.csv'
OUTER_FILE = SPLIT_DIR / 'outer_group_assignment_seed2026.csv'
SPLIT_MANIFEST_FILE = SPLIT_DIR / 'split_manifest.json'
EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'

NOTEBOOK_FILENAME = 'NB04_DEEP_LEARNING_BENCHMARK.ipynb'
RUN_REVISION = 'NB04_v1_fixed_capacity_inner_preprocessing_epoch_selection'
PACKAGE_SCHEMA = 'NIR-HUEVOS standardized result package v1.2'

MODELS = ['ANN', 'SimpleRNN', 'LSTM', 'BiLSTM']
PREPROCESSING_CANDIDATES = ['raw', 'snv', 'msc', 'sg_smooth', 'sg_deriv1']
FINAL_SEEDS = [2026, 2027, 2028]
SG_WINDOW = 11
SG_POLYORDER = 2

# Fixed a priori neural training protocol
RECURRENT_UNITS = 64
ANN_HIDDEN = [64, 32]
DENSE_AFTER_RECURRENT = 32
DROPOUT = 0.20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-3
LOSS = 'mse'
PRIMARY_SELECTION_METRIC = 'MAE_days'
REQUIRE_GPU = True
RESUME_IF_AVAILABLE = True

# GPU gate
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow:', tf.__version__)
print('GPU devices:', gpus)
if REQUIRE_GPU:
    assert len(gpus) > 0, (
        'NB04 is configured to require a GPU. In Colab choose Runtime > Change runtime type > GPU, then rerun.'
    )
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

try:
    nvidia_info = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
except Exception as e:
    nvidia_info = f'nvidia-smi unavailable: {e}'
(RESULT_DIR / 'NB04_hardware_info.txt').write_text(nvidia_info, encoding='utf-8')
print(nvidia_info.splitlines()[0] if nvidia_info else 'GPU info captured.')

# ---- Original notebook code cell 3 ----
# Integrity gate — dataset and exact frozen NB02 splits

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

split_manifest_sha = sha256_file(SPLIT_MANIFEST_FILE)
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
hash_audit.to_csv(RESULT_DIR / 'NB04_input_hash_audit.csv', index=False)

# Resume state is valid only for this exact revision + dataset + split manifest.
STATE_FILE = CHECKPOINT_DIR / 'NB04_checkpoint_state.json'
resume_valid = False
if RESUME_IF_AVAILABLE and STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        resume_valid = (
            state.get('run_revision') == RUN_REVISION and
            state.get('dataset_sha256') == dataset_sha and
            state.get('split_manifest_sha256') == split_manifest_sha
        )
    except Exception:
        resume_valid = False

if not resume_valid:
    # Clean only NB04-generated files. Keep directory itself.
    for p in RESULT_DIR.iterdir():
        if p.name != 'NB04_hardware_info.txt':
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    for p in FIG_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        'run_revision': RUN_REVISION,
        'dataset_sha256': dataset_sha,
        'split_manifest_sha256': split_manifest_sha,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'IN_PROGRESS'
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('Fresh NB04 run initialized.')
else:
    print('Valid NB04 checkpoint detected — completed work will be resumed/skipped.')

print('PASS — dataset SHA-256:', dataset_sha)
print('PASS — all frozen split hashes match NB02 manifest.')

# ---- Original notebook code cell 4 ----
# Load dataset, preserve physical wavelength order, and audit group structure

df = pd.read_csv(DATA_FILE)
outer = pd.read_csv(OUTER_FILE)

assert {'sample', 'storage_days'}.issubset(df.columns)
spectral_cols = [c for c in df.columns if c.startswith('Spectra_')]
assert len(spectral_cols) == 331, f'Expected 331 spectral variables, got {len(spectral_cols)}'

def wavelength_from_col(c):
    return float(c.replace('Spectra_', ''))

spectral_cols = sorted(spectral_cols, key=wavelength_from_col)
wavelengths = np.array([wavelength_from_col(c) for c in spectral_cols], dtype=np.float64)
X_all = df[spectral_cols].to_numpy(dtype=np.float32)
y_all = df['storage_days'].to_numpy(dtype=np.float32)

assert X_all.shape == (660, 331)
assert df['sample'].nunique() == 30
assert df['storage_days'].nunique() == 22
assert wavelengths[0] == 740 and wavelengths[-1] == 1070
assert np.all(np.diff(wavelengths) == 1)
assert set(outer['sample']) == set(df['sample'].unique())

print('X:', X_all.shape, '| eggs:', df['sample'].nunique(), '| days:', df['storage_days'].nunique())
print('Sequence axis for RNN/LSTM:', int(wavelengths[0]), '→', int(wavelengths[-1]), 'nm')

# ---- Original notebook code cell 6 ----
# Preprocessing, metrics, model builders, and reproducibility utilities

class NeuralSpectralPreprocessor:
    def __init__(self, name, sg_window=11, sg_polyorder=2):
        self.name = name
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.reference_ = None
        self.scaler_ = None

    def _base_transform(self, X):
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
                raise RuntimeError('MSC reference must be fitted on training data first.')
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

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.name == 'msc':
            self.reference_ = X.mean(axis=0)
        base = self._base_transform(X)
        self.scaler_ = StandardScaler().fit(base)
        return self

    def transform(self, X):
        if self.scaler_ is None:
            raise RuntimeError('Preprocessor must be fitted on training data first.')
        base = self._base_transform(X)
        return self.scaler_.transform(base).astype(np.float32)

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


def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_model(model_name, n_features=331):
    reg = None
    if model_name == 'ANN':
        inp = keras.Input(shape=(n_features,), name='spectrum')
        x = layers.Dense(ANN_HIDDEN[0], activation='relu')(inp)
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(ANN_HIDDEN[1], activation='relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation='linear')(x)
    else:
        inp = keras.Input(shape=(n_features, 1), name='ordered_wavelength_sequence')
        if model_name == 'SimpleRNN':
            x = layers.SimpleRNN(RECURRENT_UNITS, return_sequences=False)(inp)
        elif model_name == 'LSTM':
            x = layers.LSTM(RECURRENT_UNITS, return_sequences=False)(inp)
        elif model_name == 'BiLSTM':
            x = layers.Bidirectional(layers.LSTM(RECURRENT_UNITS, return_sequences=False))(inp)
        else:
            raise ValueError(model_name)
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(DENSE_AFTER_RECURRENT, activation='relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation='linear')(x)

    model = keras.Model(inp, out, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model


def shape_for_model(X, model_name):
    X = np.asarray(X, dtype=np.float32)
    if model_name == 'ANN':
        return X
    return X[..., np.newaxis]

# Record model complexity before any test evaluation.
complexity_rows = []
for model_name in MODELS:
    tf.keras.backend.clear_session()
    set_all_seeds(2026)
    m = build_model(model_name, len(spectral_cols))
    complexity_rows.append({
        'model': model_name,
        'parameter_count': int(m.count_params()),
        'estimated_fp32_weight_bytes': int(m.count_params() * 4),
        'input_shape': str(m.input_shape),
        'output_shape': str(m.output_shape)
    })
    del m
    gc.collect()
pd.DataFrame(complexity_rows).to_csv(RESULT_DIR / 'NB04_model_complexity.csv', index=False)
display(pd.DataFrame(complexity_rows))

# ---- Original notebook code cell 7 ----
# Exact frozen outer/inner split loaders and leakage audit

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

split_audit_rows = []
for outer_fold in range(1, 6):
    tr, te = get_outer_train_test_eggs(outer_fold)
    inner = load_inner_assignment(outer_fold)
    for inner_fold in range(1, 5):
        va = set(inner.loc[inner['inner_fold'] == inner_fold, 'sample'])
        itr = tr - va
        split_audit_rows.append({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'outer_train_eggs': len(tr),
            'outer_test_eggs': len(te),
            'inner_train_eggs': len(itr),
            'inner_val_eggs': len(va),
            'outer_overlap': len(tr & te),
            'inner_overlap': len(itr & va),
            'outer_test_in_inner': len(te & (itr | va))
        })
split_audit_df = pd.DataFrame(split_audit_rows)
assert (split_audit_df[['outer_overlap','inner_overlap','outer_test_in_inner']] == 0).all().all()
split_audit_df.to_csv(RESULT_DIR / 'NB04_split_revalidation.csv', index=False)
print('PASS — 5 outer + 20 inner frozen egg-disjoint splits revalidated.')
