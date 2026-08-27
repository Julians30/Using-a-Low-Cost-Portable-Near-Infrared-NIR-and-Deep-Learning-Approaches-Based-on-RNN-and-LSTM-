"""Public source fragment for NB06_STATISTICAL_ROBUSTNESS.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----

from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----

from pathlib import Path
from datetime import datetime, timezone
import hashlib, itertools, json, os, platform, shutil, subprocess, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')
NB03_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB03_CHEMOMETRIC_BASELINES'
NB04_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB04_DEEP_LEARNING_BENCHMARK'
NB05_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB05_WAVELENGTH_ORDER_ABLATION'

RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB06_STATISTICAL_ROBUSTNESS'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB06_STATISTICAL_ROBUSTNESS'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'

for p in [RESULT_DIR, FIG_DIR, ZIP_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

NB03_OOF = NB03_DIR / 'NB03_oof_predictions.csv'
NB03_POOLED = NB03_DIR / 'NB03_pooled_oof_metrics.csv'
NB03_PROTOCOL = NB03_DIR / 'NB03_protocol.json'
NB03_SUMMARY = NB03_DIR / 'NB03_run_summary.json'

NB04_SEEDWISE = NB04_DIR / 'NB04_oof_predictions_seedwise.csv'
NB04_POOLED = NB04_DIR / 'NB04_pooled_seedmean_metrics.csv'
NB04_PROTOCOL = NB04_DIR / 'NB04_protocol.json'
NB04_SUMMARY = NB04_DIR / 'NB04_run_summary.json'
NB04_STATUS = NB04_DIR / 'EXECUTION_STATUS.json'

NB05_SEEDWISE = NB05_DIR / 'NB05_oof_predictions_seedwise_all_orders.csv'
NB05_PROTOCOL = NB05_DIR / 'NB05_protocol.json'
NB05_SUMMARY = NB05_DIR / 'NB05_run_summary.json'
NB05_STATUS = NB05_DIR / 'EXECUTION_STATUS.json'

NOTEBOOK_FILENAME = 'NB06_STATISTICAL_ROBUSTNESS.ipynb'
RUN_REVISION = 'NB06_v1_frozen_egg_level_inference'
PACKAGE_SCHEMA = 'NIR-HUEVOS standardized result package v1.4'

EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'
EXPECTED_SPLIT_MANIFEST_SHA256 = 'fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717'

PRIMARY_MODELS = ['SVR','PLSR','ANN','BiLSTM','LSTM','SimpleRNN']
CHEM_MODELS = ['PLSR','SVR']
DEEP_MODELS = ['ANN','SimpleRNN','LSTM','BiLSTM']
FINAL_SEEDS = [2026,2027,2028]
ORDER_CONDITIONS = ['original','reversed','shuffled']

ALPHA = 0.05
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 62026

print('GPU no requerida. Este notebook corre en CPU.')

# ---- Original notebook code cell 3 ----

def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

required = [
    NB03_OOF, NB03_POOLED, NB03_PROTOCOL, NB03_SUMMARY,
    NB04_SEEDWISE, NB04_POOLED, NB04_PROTOCOL, NB04_SUMMARY, NB04_STATUS,
    NB05_SEEDWISE, NB05_PROTOCOL, NB05_SUMMARY, NB05_STATUS
]
for p in required:
    assert p.exists(), f'Falta fuente requerida: {p}'

p03 = json.loads(NB03_PROTOCOL.read_text(encoding='utf-8'))
s03 = json.loads(NB03_SUMMARY.read_text(encoding='utf-8'))
p04 = json.loads(NB04_PROTOCOL.read_text(encoding='utf-8'))
s04 = json.loads(NB04_SUMMARY.read_text(encoding='utf-8'))
e04 = json.loads(NB04_STATUS.read_text(encoding='utf-8'))
p05 = json.loads(NB05_PROTOCOL.read_text(encoding='utf-8'))
s05 = json.loads(NB05_SUMMARY.read_text(encoding='utf-8'))
e05 = json.loads(NB05_STATUS.read_text(encoding='utf-8'))

assert s03.get('status') == 'COMPLETED'
assert s04.get('status') == 'COMPLETED' and e04.get('status') == 'COMPLETED'
assert s05.get('status') == 'COMPLETED' and e05.get('status') == 'COMPLETED'

def first_key(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

for label, protocol in [('NB03',p03),('NB04',p04),('NB05',p05)]:
    ds = first_key(protocol, 'dataset_sha256', 'dataset_hash_sha256')
    if ds is not None:
        assert ds == EXPECTED_DATASET_SHA256, f'{label}: dataset hash no coincide'
    sh = first_key(protocol, 'frozen_split_manifest_sha256', 'split_manifest_sha256')
    if sh is not None:
        assert sh == EXPECTED_SPLIT_MANIFEST_SHA256, f'{label}: split hash no coincide'

audit = pd.DataFrame([{
    'source_file': str(p.relative_to(PROJECT_ROOT)),
    'size_bytes': p.stat().st_size,
    'sha256': sha256_file(p)
} for p in required])
audit.to_csv(RESULT_DIR / 'NB06_input_hash_audit.csv', index=False)
print('PASS — fuentes congeladas verificadas.')
display(audit)

# ---- Original notebook code cell 4 ----

nb03 = pd.read_csv(NB03_OOF)
nb04 = pd.read_csv(NB04_SEEDWISE)
nb05 = pd.read_csv(NB05_SEEDWISE)

assert len(nb03) == 1980
assert len(nb04) == 7920
assert len(nb05) == 23760
assert set(nb03.model) == {'DummyMean','PLSR','SVR'}
assert set(nb04.model) == set(DEEP_MODELS)
assert set(nb05.model) == set(DEEP_MODELS)
assert set(nb04.seed.astype(int)) == set(FINAL_SEEDS)
assert set(nb05.seed.astype(int)) == set(FINAL_SEEDS)
assert set(nb05.order_condition) == set(ORDER_CONDITIONS)

for m in ['DummyMean','PLSR','SVR']:
    g = nb03[nb03.model==m]
    assert len(g) == 660 and not g.duplicated(['sample','storage_days']).any()

for m in DEEP_MODELS:
    for seed in FINAL_SEEDS:
        g = nb04[(nb04.model==m) & (nb04.seed.astype(int)==seed)]
        assert len(g) == 660 and not g.duplicated(['sample','storage_days']).any()

for cond in ORDER_CONDITIONS:
    for m in DEEP_MODELS:
        for seed in FINAL_SEEDS:
            g = nb05[(nb05.order_condition==cond)&(nb05.model==m)&(nb05.seed.astype(int)==seed)]
            assert len(g) == 660 and not g.duplicated(['sample','storage_days']).any()

map03 = nb03[['sample','outer_fold']].drop_duplicates().sort_values('sample').reset_index(drop=True)
map04 = nb04[['sample','outer_fold']].drop_duplicates().sort_values('sample').reset_index(drop=True)
map05 = nb05[['sample','outer_fold']].drop_duplicates().sort_values('sample').reset_index(drop=True)
pd.testing.assert_frame_equal(map03, map04, check_dtype=False)
pd.testing.assert_frame_equal(map03, map05, check_dtype=False)

assert nb03.sample.nunique() == 30
assert nb03.storage_days.nunique() == 22
print('PASS — OOF y outer-fold coinciden entre NB03/NB04/NB05.')

# ---- Original notebook code cell 5 ----

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred,float)-np.asarray(y_true,float))))

def holm_adjust(pvalues):
    p = np.asarray(pvalues,float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    adj = np.empty(m)
    running = 0.0
    for i,val in enumerate(ps):
        running = max(running, (m-i)*val)
        adj[i] = min(1.0, running)
    out = np.empty(m)
    out[order] = adj
    return out

def rank_biserial(diff):
    d = np.asarray(diff,float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d)==0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    pos = ranks[d>0].sum()
    neg = ranks[d<0].sum()
    return float((pos-neg)/(pos+neg))

def paired_bootstrap_delta(a,b,reps=10000,seed=62026):
    d = np.asarray(a,float)-np.asarray(b,float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0,len(d),size=(reps,len(d)))
    boot = d[idx].mean(axis=1)
    lo,hi = np.quantile(boot,[0.025,0.975])
    return float(d.mean()),float(lo),float(hi)

def bootstrap_mean_ci(x,reps=10000,seed=62026):
    x = np.asarray(x,float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0,len(x),size=(reps,len(x)))
    boot = x[idx].mean(axis=1)
    lo,hi = np.quantile(boot,[0.025,0.975])
    return float(x.mean()),float(lo),float(hi)

def wilcoxon_safe(a,b):
    a = np.asarray(a,float); b = np.asarray(b,float)
    d = a-b
    if np.allclose(d,0):
        return 0.0,1.0,0
    r = stats.wilcoxon(a,b,zero_method='wilcox',alternative='two-sided',method='auto')
    return float(r.statistic),float(r.pvalue),int(np.sum(d!=0))

# ---- Original notebook code cell 6 ----

rows = []

for m in CHEM_MODELS:
    for sample,g in nb03[nb03.model==m].groupby('sample'):
        rows.append({'sample':int(sample),'model':m,'MAE_days':mae(g.storage_days,g.y_pred),
                     'aggregation_rule':'deterministic OOF'})

deep_seed_rows = []
for (m,seed,sample),g in nb04.groupby(['model','seed','sample']):
    deep_seed_rows.append({'model':m,'seed':int(seed),'sample':int(sample),
                           'MAE_days':mae(g.storage_days,g.y_pred)})
deep_seed = pd.DataFrame(deep_seed_rows)
deep_egg = deep_seed.groupby(['model','sample'],as_index=False).agg(
    MAE_days=('MAE_days','mean'),
    seed_SD_MAE_days=('MAE_days','std')
)
for _,r in deep_egg.iterrows():
    rows.append({'sample':int(r['sample']),'model':r['model'],'MAE_days':float(r['MAE_days']),
                 'aggregation_rule':'mean of 3 seed-specific per-egg MAEs'})

primary_long = pd.DataFrame(rows)
primary_wide = primary_long.pivot(index='sample',columns='model',values='MAE_days')[PRIMARY_MODELS]

assert primary_long.shape[0] == 180
assert primary_wide.shape == (30,6)
assert primary_wide.notna().all().all()

primary_long.to_csv(RESULT_DIR/'NB06_primary_per_egg_MAE_long.csv',index=False)
primary_wide.reset_index().to_csv(RESULT_DIR/'NB06_primary_per_egg_MAE_wide.csv',index=False)
deep_seed.to_csv(RESULT_DIR/'NB06_deep_per_egg_MAE_seedwise.csv',index=False)

rank_matrix = primary_wide.rank(axis=1,ascending=True,method='average')
summary = []
for i,m in enumerate(PRIMARY_MODELS):
    x = primary_wide[m].to_numpy()
    mean_x,lo,hi = bootstrap_mean_ci(x,BOOTSTRAP_REPS,BOOTSTRAP_SEED+i)
    summary.append({
        'model':m,
        'mean_per_egg_MAE_days':mean_x,
        'bootstrap95_CI_low':lo,
        'bootstrap95_CI_high':hi,
        'SD_per_egg_MAE_days':float(np.std(x,ddof=1)),
        'median_per_egg_MAE_days':float(np.median(x)),
        'IQR_per_egg_MAE_days':float(np.quantile(x,.75)-np.quantile(x,.25)),
        'average_rank_lower_is_better':float(rank_matrix[m].mean())
    })
primary_summary = pd.DataFrame(summary).sort_values('mean_per_egg_MAE_days')
primary_summary.to_csv(RESULT_DIR/'NB06_primary_model_summary.csv',index=False)
rank_matrix.reset_index().to_csv(RESULT_DIR/'NB06_per_egg_model_ranks.csv',index=False)
display(primary_summary)

# ---- Original notebook code cell 7 ----

fr = stats.friedmanchisquare(*[primary_wide[m].to_numpy() for m in PRIMARY_MODELS])
n = len(primary_wide); k = len(PRIMARY_MODELS)
kendall_w = float(fr.statistic/(n*(k-1)))

friedman_table = pd.DataFrame([{
    'analysis':'Primary model comparison on per-egg MAE',
    'n_eggs':n,'k_models':k,'friedman_chi2':float(fr.statistic),
    'df':k-1,'p_value':float(fr.pvalue),'kendall_W':kendall_w,
    'alpha':ALPHA,'significant':bool(fr.pvalue<ALPHA)
}])
friedman_table.to_csv(RESULT_DIR/'NB06_friedman_primary.csv',index=False)
display(friedman_table)

pair_rows = []
for j,(a_name,b_name) in enumerate(itertools.combinations(PRIMARY_MODELS,2)):
    a = primary_wide[a_name].to_numpy()
    b = primary_wide[b_name].to_numpy()
    stat,p_raw,nz = wilcoxon_safe(a,b)
    dmean,lo,hi = paired_bootstrap_delta(a,b,BOOTSTRAP_REPS,BOOTSTRAP_SEED+100+j)
    diff = a-b
    pair_rows.append({
        'model_A':a_name,'model_B':b_name,
        'delta_MAE_A_minus_B_days':dmean,
        'delta95_CI_low':lo,'delta95_CI_high':hi,
        'median_delta_A_minus_B_days':float(np.median(diff)),
        'wilcoxon_statistic':stat,'p_raw':p_raw,'n_nonzero_pairs':nz,
        'rank_biserial_A_minus_B':rank_biserial(diff)
    })

pairwise = pd.DataFrame(pair_rows)
pairwise['p_holm'] = holm_adjust(pairwise.p_raw.to_numpy())
pairwise['significant_holm_0.05'] = pairwise.p_holm < ALPHA
pairwise['CI_excludes_zero'] = (pairwise.delta95_CI_high<0)|(pairwise.delta95_CI_low>0)
pairwise['top3_pair'] = pairwise.apply(
    lambda r: r.model_A in {'SVR','PLSR','ANN'} and r.model_B in {'SVR','PLSR','ANN'},axis=1
)
pairwise.to_csv(RESULT_DIR/'NB06_pairwise_wilcoxon_holm.csv',index=False)
display(pairwise.sort_values(['p_holm','p_raw']))
print('TOP-3')
display(pairwise[pairwise.top3_pair])

# ---- Original notebook code cell 8 ----

dummy = nb03[nb03.model=='DummyMean'].groupby('sample').apply(
    lambda g: mae(g.storage_days,g.y_pred)
)
dummy.index = dummy.index.astype(int)

drows = []
for i,m in enumerate(PRIMARY_MODELS):
    a = primary_wide[m].to_numpy()
    b = dummy.loc[primary_wide.index].to_numpy()
    stat,p_raw,nz = wilcoxon_safe(a,b)
    dmean,lo,hi = paired_bootstrap_delta(a,b,BOOTSTRAP_REPS,BOOTSTRAP_SEED+1000+i)
    drows.append({
        'model':m,'DummyMean_MAE_days_per_egg':float(np.mean(b)),
        'model_mean_per_egg_MAE_days':float(np.mean(a)),
        'delta_model_minus_Dummy_days':dmean,
        'delta95_CI_low':lo,'delta95_CI_high':hi,
        'wilcoxon_statistic':stat,'p_raw':p_raw,
        'rank_biserial_model_minus_Dummy':rank_biserial(a-b)
    })
vs_dummy = pd.DataFrame(drows)
vs_dummy['p_holm'] = holm_adjust(vs_dummy.p_raw.to_numpy())
vs_dummy['significant_holm_0.05'] = vs_dummy.p_holm < ALPHA
vs_dummy.to_csv(RESULT_DIR/'NB06_vs_dummy_secondary.csv',index=False)
display(vs_dummy.sort_values('model_mean_per_egg_MAE_days'))
