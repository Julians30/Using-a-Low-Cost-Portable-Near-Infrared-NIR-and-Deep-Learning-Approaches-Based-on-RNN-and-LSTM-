"""Public source fragment for NB07_PRACTICAL_APPLICABILITY.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 12 ----

def parse_hyperparameters(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value,float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            out = pyast.literal_eval(text)
            return out if isinstance(out,dict) else {}
        except Exception:
            return {}

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'No se encontró ninguna de estas columnas: {candidates}; disponibles={df.columns.tolist()}')
    return None

model_col03 = find_col(cfg03, ['model','Model'])
prep_col03 = find_col(cfg03, ['selected_preprocessing','preprocessing','prep'])
hp_col03 = find_col(cfg03, ['selected_hyperparameters','hyperparameters','best_params'], required=False)

model_col04 = find_col(cfg04, ['model','Model'])
prep_col04 = find_col(cfg04, ['selected_preprocessing','preprocessing','prep'])

def deterministic_mode(series):
    counts = series.astype(str).value_counts()
    max_count = counts.max()
    return sorted(counts[counts == max_count].index.tolist())[0]

def hp_dicts_for(df, model_col, model_name, hp_col):
    rows = df[df[model_col].astype(str) == model_name]
    if hp_col is not None:
        return [parse_hyperparameters(v) for v in rows[hp_col].tolist()]
    dicts = []
    for _, r in rows.iterrows():
        d = {}
        for k in ['n_components','C','epsilon','gamma']:
            if k in df.columns and pd.notna(r[k]):
                d[k] = float(r[k])
        dicts.append(d)
    return dicts

def collect_param(dicts, key):
    vals = [d[key] for d in dicts if key in d and d[key] is not None]
    if not vals:
        raise KeyError(f'No se pudo recuperar {key} de las configuraciones congeladas.')
    return np.asarray(vals, dtype=float)

rep_cfg = {}

# PLSR
pls_rows = cfg03[cfg03[model_col03].astype(str) == 'PLSR']
pls_prep = deterministic_mode(pls_rows[prep_col03])
pls_hps = hp_dicts_for(cfg03, model_col03, 'PLSR', hp_col03)
pls_ncomp_vals = collect_param(pls_hps, 'n_components')
pls_ncomp = int(np.median(pls_ncomp_vals))

rep_cfg['PLSR'] = {
    'preprocessing': pls_prep,
    'n_components': pls_ncomp,
    'selection_rule': 'mode preprocessing + median selected n_components across 5 outer folds'
}

# SVR
svr_rows = cfg03[cfg03[model_col03].astype(str) == 'SVR']
svr_prep = deterministic_mode(svr_rows[prep_col03])
svr_hps = hp_dicts_for(cfg03, model_col03, 'SVR', hp_col03)

svr_C = float(np.median(collect_param(svr_hps, 'C')))
svr_eps = float(np.median(collect_param(svr_hps, 'epsilon')))
svr_gamma = float(np.median(collect_param(svr_hps, 'gamma')))

rep_cfg['SVR'] = {
    'preprocessing': svr_prep,
    'C': svr_C,
    'epsilon': svr_eps,
    'gamma': svr_gamma,
    'selection_rule': 'mode preprocessing + componentwise median selected hyperparameters across 5 outer folds'
}

# DL: solo preprocessing representativo para benchmark end-to-end.
for model_name in DEEP_MODELS:
    rows = cfg04[cfg04[model_col04].astype(str) == model_name]
    assert len(rows) > 0, f'No hay configuraciones NB04 para {model_name}'
    rep_cfg[model_name] = {
        'preprocessing': deterministic_mode(rows[prep_col04]),
        'selection_rule': 'mode selected preprocessing across 5 outer folds; architecture fixed a priori'
    }

rep_cfg_df = pd.DataFrame([
    {'model': k, **v} for k,v in rep_cfg.items()
])
rep_cfg_df.to_csv(RESULT_DIR / 'NB07_representative_deployment_configurations.csv', index=False)

print(json.dumps(rep_cfg, indent=2))

# ---- Original notebook code cell 13 ----

class SpectralPreprocessor:
    def __init__(self, method, sg_window=11, sg_polyorder=2):
        self.method = str(method).lower()
        self.sg_window = int(sg_window)
        self.sg_polyorder = int(sg_polyorder)
        self.msc_reference_ = None
        self.scaler_ = None

    def _base_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.method == 'msc':
            self.msc_reference_ = X.mean(axis=0)
        return self._base_transform(X)

    def _base_transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.method in ['raw','none']:
            return X.copy()

        if self.method == 'snv':
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            return (X - mu) / sd

        if self.method == 'msc':
            if self.msc_reference_ is None:
                raise RuntimeError('MSC reference not fitted.')
            ref = self.msc_reference_
            ref_c = ref - ref.mean()
            ref_ss = np.sum(ref_c**2)
            x_mean = X.mean(axis=1, keepdims=True)
            slopes = np.sum((X - x_mean) * ref_c[None,:], axis=1) / ref_ss
            slopes = np.where(np.abs(slopes) < 1e-12, 1.0, slopes)
            intercepts = X.mean(axis=1) - slopes * ref.mean()
            return (X - intercepts[:,None]) / slopes[:,None]

        if self.method == 'sg_smooth':
            return savgol_filter(
                X, window_length=self.sg_window, polyorder=self.sg_polyorder,
                deriv=0, axis=1, mode='interp'
            )

        if self.method == 'sg_deriv1':
            return savgol_filter(
                X, window_length=self.sg_window, polyorder=self.sg_polyorder,
                deriv=1, delta=1.0, axis=1, mode='interp'
            )

        raise ValueError(f'Preprocesamiento desconocido: {self.method}')

    def fit(self, X):
        Xb = self._base_fit(X)
        self.scaler_ = StandardScaler().fit(Xb)
        return self

    def transform(self, X):
        if self.scaler_ is None:
            raise RuntimeError('Scaler not fitted.')
        Xb = self._base_transform(X)
        return self.scaler_.transform(Xb)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Sanity checks de preprocesamiento.
for method in sorted(set(v['preprocessing'] for v in rep_cfg.values())):
    pp = SpectralPreprocessor(method).fit(X)
    Z = pp.transform(X[:5])
    assert Z.shape == (5,331)
    assert np.isfinite(Z).all()
print('PASS — preprocessors de despliegue reconstruidos.')

# ---- Original notebook code cell 14 ----

# Refit PLSR/SVR en los 30 huevos SOLO para tamaño/latencia/ingeniería.
# No se calcula ni reporta training score.

engineering_models = {}
engineering_preprocessors = {}
engineering_rows = []

for model_name in ['PLSR','SVR']:
    cfg = rep_cfg[model_name]
    pp = SpectralPreprocessor(cfg['preprocessing']).fit(X)
    Xp = pp.transform(X)

    if model_name == 'PLSR':
        estimator = PLSRegression(
            n_components=int(cfg['n_components']),
            scale=False,
            max_iter=500
        )
    else:
        estimator = SVR(
            kernel='rbf',
            C=float(cfg['C']),
            epsilon=float(cfg['epsilon']),
            gamma=float(cfg['gamma'])
        )

    t0 = time.perf_counter()
    estimator.fit(Xp, y)
    fit_seconds = time.perf_counter() - t0

    engineering_models[model_name] = estimator
    engineering_preprocessors[model_name] = pp

    tmp_path = RESULT_DIR / f'_TEMP_{model_name}_deployment.joblib'
    joblib.dump({'preprocessor':pp,'model':estimator,'config':cfg}, tmp_path)
    size_mb = tmp_path.stat().st_size / (1024**2)

    if model_name == 'PLSR':
        complexity_name = 'n_components'
        complexity_value = int(cfg['n_components'])
    else:
        complexity_name = 'n_support_vectors'
        complexity_value = int(len(estimator.support_))

    engineering_rows.append({
        'model': model_name,
        'engineering_fit_on_all_660_spectra': True,
        'generalization_performance_from_this_fit_reported': False,
        'representative_preprocessing': cfg['preprocessing'],
        'complexity_measure': complexity_name,
        'complexity_value': complexity_value,
        'serialized_size_MB': float(size_mb),
        'final_refit_seconds_CPU': float(fit_seconds)
    })

    tmp_path.unlink(missing_ok=True)

print('PASS — PLSR/SVR refit final realizado únicamente para ingeniería de despliegue.')

# ---- Original notebook code cell 15 ----

# Reconstrucción de las arquitecturas congeladas NB04.
import tensorflow as tf

tf.keras.backend.clear_session()
tf.random.set_seed(LATENCY_SEED)
np.random.seed(LATENCY_SEED)

model_specs = p04['model_specs']
assert p04['architecture_hyperparameters'].startswith('fixed')

def build_deep_model(model_name):
    if model_name == 'ANN':
        inp = tf.keras.Input(shape=(331,), name='spectral_vector')
        x = tf.keras.layers.Dense(64, activation='relu')(inp)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp,out,name='ANN')

    inp = tf.keras.Input(shape=(331,1), name='ordered_wavelength_sequence')

    if model_name == 'SimpleRNN':
        x = tf.keras.layers.SimpleRNN(64)(inp)
    elif model_name == 'LSTM':
        x = tf.keras.layers.LSTM(64)(inp)
    elif model_name == 'BiLSTM':
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(inp)
    else:
        raise ValueError(model_name)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp,out,name=model_name)

deep_arch_models = {}

for model_name in DEEP_MODELS:
    with tf.device('/CPU:0'):
        model = build_deep_model(model_name)
        _ = model(
            tf.zeros((1,331) if model_name=='ANN' else (1,331,1), dtype=tf.float32),
            training=False
        )

    deep_arch_models[model_name] = model

    tmp_path = RESULT_DIR / f'_TEMP_{model_name}_architecture.keras'
    model.save(tmp_path, include_optimizer=False)
    size_mb = tmp_path.stat().st_size / (1024**2)

    engineering_rows.append({
        'model': model_name,
        'engineering_fit_on_all_660_spectra': False,
        'generalization_performance_from_this_fit_reported': False,
        'representative_preprocessing': rep_cfg[model_name]['preprocessing'],
        'complexity_measure': 'trainable_parameters',
        'complexity_value': int(model.count_params()),
        'serialized_size_MB': float(size_mb),
        'final_refit_seconds_CPU': np.nan
    })

    tmp_path.unlink(missing_ok=True)

engineering_complexity = pd.DataFrame(engineering_rows)
engineering_complexity.to_csv(RESULT_DIR / 'NB07_model_complexity_and_size.csv', index=False)

# Si NB04 guardó su propia tabla de complejidad, la preservamos como evidencia auxiliar.
if NB04_COMPLEXITY.exists():
    nb04_complexity = pd.read_csv(NB04_COMPLEXITY)
    nb04_complexity.to_csv(RESULT_DIR / 'NB07_source_NB04_model_complexity.csv', index=False)
    print('Tabla original NB04_model_complexity.csv:')
    display(nb04_complexity)

display(engineering_complexity)

# ---- Original notebook code cell 16 ----

# Hardware para interpretar latencia.
hardware_lines = []
hardware_lines.append(f'Python: {sys.version}')
hardware_lines.append(f'Platform: {platform.platform()}')
hardware_lines.append(f'Processor: {platform.processor()}')
hardware_lines.append(f'TensorFlow: {tf.__version__}')
hardware_lines.append(f'TF physical CPUs: {tf.config.list_physical_devices("CPU")}')
hardware_lines.append(f'TF physical GPUs: {tf.config.list_physical_devices("GPU")}')
try:
    hardware_lines.append('\nLSCPU:\n' + subprocess.check_output(['lscpu'], text=True))
except Exception as e:
    hardware_lines.append(f'lscpu unavailable: {e}')

hardware_text = '\n'.join(hardware_lines)
(RESULT_DIR / 'NB07_latency_hardware_info.txt').write_text(hardware_text, encoding='utf-8')
print(hardware_text[:4000])
