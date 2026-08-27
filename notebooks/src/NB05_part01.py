"""Public source fragment for NB05_WAVELENGTH_ORDER_ABLATION.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----

from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----

# Imports, paths, fixed protocol, and GPU gate
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, os, platform, random, shutil, subprocess, sys, time, warnings

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
NB04_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB04_DEEP_LEARNING_BENCHMARK'
RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB05_WAVELENGTH_ORDER_ABLATION'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB05_WAVELENGTH_ORDER_ABLATION'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'
CHECKPOINT_DIR = RESULT_DIR / '_CHECKPOINT'

for p in [RESULT_DIR, FIG_DIR, ZIP_DIR, NOTEBOOKS_DIR, CHECKPOINT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATA_FILE = RAW_DIR / 'dataset_egg_storage_RAW.csv'
OUTER_FILE = SPLIT_DIR / 'outer_group_assignment_seed2026.csv'
SPLIT_MANIFEST_FILE = SPLIT_DIR / 'split_manifest.json'
NB04_PROTOCOL_FILE = NB04_DIR / 'NB04_protocol.json'
NB04_SUMMARY_FILE = NB04_DIR / 'NB04_run_summary.json'
NB04_SELECTED_FILE = NB04_DIR / 'NB04_selected_configurations.csv'
NB04_PRED_FILE = NB04_DIR / 'NB04_oof_predictions_seedwise.csv'

EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'
EXPECTED_SPLIT_MANIFEST_SHA256 = 'fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717'
EXPECTED_NB04_REVISION = 'NB04_v1_fixed_capacity_inner_preprocessing_epoch_selection'

NOTEBOOK_FILENAME = 'NB05_WAVELENGTH_ORDER_ABLATION.ipynb'
RUN_REVISION = 'NB05_v1_frozen_NB04_recipe_order_ablation'
PACKAGE_SCHEMA = 'NIR-HUEVOS standardized result package v1.3'

MODELS = ['ANN', 'SimpleRNN', 'LSTM', 'BiLSTM']
FINAL_SEEDS = [2026, 2027, 2028]
ORDER_CONDITIONS = ['original', 'reversed', 'shuffled']
TRAIN_CONDITIONS = ['reversed', 'shuffled']
SHUFFLE_SEED = 52026

# Exact NB04 architecture/training constants
RECURRENT_UNITS = 64
ANN_HIDDEN = [64, 32]
DENSE_AFTER_RECURRENT = 32
DROPOUT = 0.20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LOSS = 'mse'
SG_WINDOW = 11
SG_POLYORDER = 2

REQUIRE_GPU = True
RESUME_IF_AVAILABLE = True

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow:', tf.__version__)
print('GPU devices:', gpus)
if REQUIRE_GPU:
    assert len(gpus) > 0, (
        'NB05 requires a GPU. In Colab choose Runtime > Change runtime type > GPU, then rerun.'
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
(RESULT_DIR / 'NB05_hardware_info.txt').write_text(nvidia_info, encoding='utf-8')
print(nvidia_info.splitlines()[0] if nvidia_info else 'GPU info captured.')

# ---- Original notebook code cell 3 ----

# Integrity gate: dataset, frozen splits, and approved NB04 evidence

def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

required = [
    DATA_FILE, OUTER_FILE, SPLIT_MANIFEST_FILE,
    NB04_PROTOCOL_FILE, NB04_SUMMARY_FILE, NB04_SELECTED_FILE, NB04_PRED_FILE
]
for p in required:
    assert p.exists(), f'Missing required input: {p}'

dataset_sha = sha256_file(DATA_FILE)
split_manifest_sha = sha256_file(SPLIT_MANIFEST_FILE)
assert dataset_sha == EXPECTED_DATASET_SHA256, 'Dataset hash changed.'
assert split_manifest_sha == EXPECTED_SPLIT_MANIFEST_SHA256, 'Frozen split manifest hash changed.'

split_manifest = json.loads(SPLIT_MANIFEST_FILE.read_text(encoding='utf-8'))
for file_name, expected_hash in split_manifest['files'].items():
    p = SPLIT_DIR / file_name
    assert p.exists(), f'Missing frozen split file: {p}'
    assert sha256_file(p) == expected_hash, f'Frozen split changed: {file_name}'

nb04_protocol = json.loads(NB04_PROTOCOL_FILE.read_text(encoding='utf-8'))
nb04_summary = json.loads(NB04_SUMMARY_FILE.read_text(encoding='utf-8'))
assert nb04_protocol['run_revision'] == EXPECTED_NB04_REVISION
assert nb04_summary['status'] == 'COMPLETED'
assert nb04_protocol['dataset_sha256'] == dataset_sha
assert nb04_protocol['frozen_split_manifest_sha256'] == split_manifest_sha
assert nb04_summary['total_seedwise_oof_predictions'] == 7920

selected_nb04 = pd.read_csv(NB04_SELECTED_FILE)
original_nb04 = pd.read_csv(NB04_PRED_FILE)
assert len(selected_nb04) == 20
assert len(original_nb04) == 7920
assert not selected_nb04.duplicated(['outer_fold','model']).any()
assert not original_nb04.duplicated(['outer_fold','model','seed','sample','storage_days']).any()

# Checkpoint identity
STATE_FILE = CHECKPOINT_DIR / 'NB05_checkpoint_state.json'
resume_valid = False
if RESUME_IF_AVAILABLE and STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        resume_valid = (
            state.get('run_revision') == RUN_REVISION and
            state.get('dataset_sha256') == dataset_sha and
            state.get('split_manifest_sha256') == split_manifest_sha and
            state.get('nb04_revision') == EXPECTED_NB04_REVISION and
            state.get('shuffle_seed') == SHUFFLE_SEED
        )
    except Exception:
        resume_valid = False

if not resume_valid:
    for p in RESULT_DIR.iterdir():
        if p.name not in ['NB05_hardware_info.txt', '_CHECKPOINT']:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    for p in FIG_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    if CHECKPOINT_DIR.exists():
        for p in CHECKPOINT_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    state = {
        'run_revision': RUN_REVISION,
        'dataset_sha256': dataset_sha,
        'split_manifest_sha256': split_manifest_sha,
        'nb04_revision': EXPECTED_NB04_REVISION,
        'shuffle_seed': SHUFFLE_SEED,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'IN_PROGRESS'
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('Fresh NB05 run initialized.')
else:
    print('Valid NB05 checkpoint detected — completed ablation fits will be skipped.')

print('PASS — dataset, frozen splits, and approved NB04 evidence verified.')

# ---- Original notebook code cell 4 ----

# Load data in physical wavelength order and construct fixed order maps

df = pd.read_csv(DATA_FILE)
outer = pd.read_csv(OUTER_FILE)

spectral_cols = [c for c in df.columns if c.startswith('Spectra_')]
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

N_FEATURES = len(spectral_cols)
ORIGINAL_IDX = np.arange(N_FEATURES, dtype=int)
REVERSED_IDX = ORIGINAL_IDX[::-1].copy()
SHUFFLED_IDX = np.random.default_rng(SHUFFLE_SEED).permutation(N_FEATURES)

assert np.array_equal(np.sort(SHUFFLED_IDX), ORIGINAL_IDX)
assert not np.array_equal(SHUFFLED_IDX, ORIGINAL_IDX)
assert not np.array_equal(SHUFFLED_IDX, REVERSED_IDX)

order_rows = []
for condition, idx in [
    ('original', ORIGINAL_IDX),
    ('reversed', REVERSED_IDX),
    ('shuffled', SHUFFLED_IDX)
]:
    for new_position, source_index in enumerate(idx):
        order_rows.append({
            'condition': condition,
            'new_position_0based': int(new_position),
            'source_index_0based': int(source_index),
            'wavelength_nm_at_new_position': float(wavelengths[source_index])
        })
order_map = pd.DataFrame(order_rows)
order_map.to_csv(RESULT_DIR / 'NB05_wavelength_order_map.csv', index=False)

print('Original first/last:', wavelengths[ORIGINAL_IDX[0]], wavelengths[ORIGINAL_IDX[-1]])
print('Reversed first/last:', wavelengths[REVERSED_IDX[0]], wavelengths[REVERSED_IDX[-1]])
print('Shuffled first 12 wavelengths:', wavelengths[SHUFFLED_IDX[:12]].astype(int).tolist())

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
            raise RuntimeError('Preprocessor must be fitted first.')
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
    if model_name == 'ANN':
        inp = keras.Input(shape=(n_features,), name='spectrum')
        x = layers.Dense(ANN_HIDDEN[0], activation='relu')(inp)
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(ANN_HIDDEN[1], activation='relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation='linear')(x)
    else:
        inp = keras.Input(shape=(n_features, 1), name='wavelength_sequence')
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
    return X if model_name == 'ANN' else X[..., np.newaxis]


def apply_order(X, condition):
    if condition == 'original':
        return X[:, ORIGINAL_IDX]
    if condition == 'reversed':
        return X[:, REVERSED_IDX]
    if condition == 'shuffled':
        return X[:, SHUFFLED_IDX]
    raise ValueError(condition)


def get_outer_train_test_eggs(outer_fold):
    test_eggs = set(outer.loc[outer['outer_fold'] == outer_fold, 'sample'].tolist())
    train_eggs = set(df['sample'].unique()) - test_eggs
    assert len(train_eggs) == 24 and len(test_eggs) == 6
    assert not (train_eggs & test_eggs)
    return train_eggs, test_eggs
