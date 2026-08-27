"""Public source fragment for NB08_PUBLICATION_FIGURES_TABLES.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 1 ----

from google.colab import drive
drive.mount('/content/drive')

# ---- Original notebook code cell 2 ----

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, shutil, subprocess, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path('/content/drive/MyDrive/NIR_HUEVOS_PAPER_REBUILD_2026')

NB03_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB03_CHEMOMETRIC_BASELINES'
NB04_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB04_DEEP_LEARNING_BENCHMARK'
NB06_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB06_STATISTICAL_ROBUSTNESS'
NB07_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB07_PRACTICAL_APPLICABILITY'

RESULT_DIR = PROJECT_ROOT / '05_RESULTS' / 'NB08_PUBLICATION_FIGURES_TABLES'
FIG_DIR = PROJECT_ROOT / '06_FIGURES' / 'NB08_PUBLICATION_FIGURES_TABLES'
TABLE_DIR = PROJECT_ROOT / '07_TABLES' / 'NB08_PUBLICATION_FIGURES_TABLES'
ZIP_DIR = PROJECT_ROOT / '05_RESULTS' / 'ZIP_PACKAGES'
NOTEBOOKS_DIR = PROJECT_ROOT / '04_NOTEBOOKS'

for p in [RESULT_DIR, FIG_DIR, TABLE_DIR, ZIP_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

NB03_SELECTED = NB03_DIR / 'NB03_selected_configurations.csv'
NB04_SELECTED = NB04_DIR / 'NB04_selected_configurations.csv'
NB04_PROTOCOL = NB04_DIR / 'NB04_protocol.json'

NB06_PRIMARY_WIDE = NB06_DIR / 'NB06_primary_per_egg_MAE_wide.csv'
NB06_SUMMARY = NB06_DIR / 'NB06_primary_model_summary.csv'
NB06_FRIEDMAN = NB06_DIR / 'NB06_friedman_primary.csv'
NB06_PAIRWISE = NB06_DIR / 'NB06_pairwise_wilcoxon_holm.csv'
NB06_ORDER_EGG = NB06_DIR / 'NB06_order_per_egg_MAE_primary.csv'
NB06_ORDER_FRIEDMAN = NB06_DIR / 'NB06_order_friedman_by_model.csv'
NB06_ORDER_PAIRWISE = NB06_DIR / 'NB06_order_pairwise_wilcoxon_holm.csv'

NB07_OOF = NB07_DIR / 'NB07_oof_operational_predictions.csv'
NB07_OPERATIONAL = NB07_DIR / 'NB07_operational_performance_summary.csv'
NB07_PHASE = NB07_DIR / 'NB07_metrics_by_storage_phase.csv'
NB07_DAY = NB07_DIR / 'NB07_metrics_by_storage_day.csv'
NB07_CLIPPING = NB07_DIR / 'NB07_operational_clipping_sensitivity.csv'
NB07_COMPLEXITY = NB07_DIR / 'NB07_model_complexity_and_size.csv'
NB07_LATENCY = NB07_DIR / 'NB07_CPU_inference_latency.csv'
NB07_PRACTICAL = NB07_DIR / 'NB07_integrated_practical_applicability_table.csv'
NB07_STATUS = NB07_DIR / 'EXECUTION_STATUS.json'

NOTEBOOK_FILENAME = 'NB08_PUBLICATION_FIGURES_TABLES.ipynb'
RUN_REVISION = 'NB08_v1_MDPI_publication_ready_600dpi'
PACKAGE_SCHEMA = 'NIR-HUEVOS standardized result package v1.6'

EXPECTED_DATASET_SHA256 = 'cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e'
EXPECTED_SPLIT_MANIFEST_SHA256 = 'fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717'

MODELS_MAIN = ['SVR','PLSR','ANN','BiLSTM','LSTM','SimpleRNN']
TOP3 = ['SVR','PLSR','ANN']
DEEP_MODELS = ['ANN','SimpleRNN','LSTM','BiLSTM']
ORDER_CONDITIONS = ['original','reversed','shuffled']

# MDPI-inspired publication settings.
FIG_DPI = 600
BASE_FONT_SIZE = 10

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_candidates = ['Palatino Linotype','Palatino','Book Antiqua','URW Palladio L','DejaVu Serif']
FONT_FAMILY = next((f for f in font_candidates if f in available_fonts), 'DejaVu Serif')

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': BASE_FONT_SIZE,
    'axes.labelsize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE,
    'ytick.labelsize': BASE_FONT_SIZE,
    'legend.fontsize': BASE_FONT_SIZE,
    'figure.dpi': 120,
    'savefig.dpi': FIG_DPI,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print('Figure font:', FONT_FAMILY)
print('Figure text size:', BASE_FONT_SIZE, 'pt')
print('Export resolution:', FIG_DPI, 'dpi')

# ---- Original notebook code cell 3 ----

def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

required = [
    NB03_SELECTED, NB04_SELECTED, NB04_PROTOCOL,
    NB06_PRIMARY_WIDE, NB06_SUMMARY, NB06_FRIEDMAN, NB06_PAIRWISE,
    NB06_ORDER_EGG, NB06_ORDER_FRIEDMAN, NB06_ORDER_PAIRWISE,
    NB07_OOF, NB07_OPERATIONAL, NB07_PHASE, NB07_DAY, NB07_CLIPPING,
    NB07_COMPLEXITY, NB07_LATENCY, NB07_PRACTICAL, NB07_STATUS
]

for p in required:
    assert p.exists(), f'Missing required frozen source: {p}'

status07 = json.loads(NB07_STATUS.read_text(encoding='utf-8'))
assert status07.get('status') == 'COMPLETED'

audit = pd.DataFrame([{
    'source_file': str(p.relative_to(PROJECT_ROOT)),
    'size_bytes': p.stat().st_size,
    'sha256': sha256_file(p)
} for p in required])
audit.to_csv(RESULT_DIR / 'NB08_input_hash_audit.csv', index=False)

print('PASS — frozen NB03/NB04/NB06/NB07 inputs verified.')
display(audit)

# ---- Original notebook code cell 4 ----

# Load frozen evidence.
cfg03 = pd.read_csv(NB03_SELECTED)
cfg04 = pd.read_csv(NB04_SELECTED)
p04 = json.loads(NB04_PROTOCOL.read_text(encoding='utf-8'))

mae_wide = pd.read_csv(NB06_PRIMARY_WIDE)
model_summary = pd.read_csv(NB06_SUMMARY)
friedman = pd.read_csv(NB06_FRIEDMAN)
pairwise = pd.read_csv(NB06_PAIRWISE)
order_egg = pd.read_csv(NB06_ORDER_EGG)
order_friedman = pd.read_csv(NB06_ORDER_FRIEDMAN)
order_pairwise = pd.read_csv(NB06_ORDER_PAIRWISE)

oof = pd.read_csv(NB07_OOF)
operational = pd.read_csv(NB07_OPERATIONAL)
phase = pd.read_csv(NB07_PHASE)
day = pd.read_csv(NB07_DAY)
clipping = pd.read_csv(NB07_CLIPPING)
complexity = pd.read_csv(NB07_COMPLEXITY)
latency = pd.read_csv(NB07_LATENCY)
practical = pd.read_csv(NB07_PRACTICAL)

assert mae_wide.shape == (30,7)  # sample + 6 models
assert oof.groupby('model').size().eq(660).all()
assert len(operational) == 7
assert len(order_egg) == 360

print('PASS — publication datasets loaded.')

# ---- Original notebook code cell 5 ----

# Helpers: no figure titles; save both PNG and TIFF at 600 dpi.

def save_pubfig(fig, stem):
    png = FIG_DIR / f'{stem}.png'
    tif = FIG_DIR / f'{stem}.tiff'
    fig.savefig(png, dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(tif, dpi=FIG_DPI, bbox_inches='tight', facecolor='white', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    return png, tif

def panel_label(ax, label):
    ax.text(
        0.01, 0.99, label,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=BASE_FONT_SIZE,
        fontweight='bold'
    )

def model_tag(ax, text):
    ax.text(
        0.98, 0.04, text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=BASE_FONT_SIZE
    )

# ---- Original notebook code cell 6 ----

# Figure 1 — leakage-safe study workflow. NO title inside the figure.

fig, ax = plt.subplots(figsize=(12,5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

steps = [
    (0.02, 0.60, 0.14, 0.22, '30 shell eggs\n22 storage days'),
    (0.19, 0.60, 0.14, 0.22, '660 SCiO NIR spectra\n740–1070 nm\n331 wavelengths'),
    (0.36, 0.60, 0.14, 0.22, 'Frozen outer CV\n5 egg-disjoint folds\n24 train / 6 test eggs'),
    (0.53, 0.60, 0.14, 0.22, 'Inner selection\n4 egg-disjoint folds\ntrain-only tuning'),
    (0.70, 0.60, 0.14, 0.22, 'PLSR · SVR · ANN\nSimpleRNN · LSTM\nBiLSTM'),
    (0.85, 0.60, 0.13, 0.22, 'OOF predictions\n30 unseen eggs')
]

for x,y,w,h,text in steps:
    box = FancyBboxPatch(
        (x,y), w,h,
        boxstyle='round,pad=0.012,rounding_size=0.012',
        linewidth=1.0, edgecolor='black', facecolor='white'
    )
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=BASE_FONT_SIZE)

for i in range(len(steps)-1):
    x,y,w,h,_ = steps[i]
    xn,yn,wn,hn,_ = steps[i+1]
    arrow = FancyArrowPatch(
        (x+w, y+h/2), (xn, yn+hn/2),
        arrowstyle='-|>', mutation_scale=12,
        linewidth=1.0
    )
    ax.add_patch(arrow)

bottom = [
    (0.18,0.18,0.18,0.20,'Statistical robustness\nEgg-level MAE\nFriedman · Wilcoxon-Holm\nCluster bootstrap'),
    (0.41,0.18,0.18,0.20,'Wavelength-order ablation\nOriginal · Reversed · Shuffled'),
    (0.64,0.18,0.18,0.20,'Practical applicability\nTolerance · phase error\nmodel size · CPU latency')
]
for x,y,w,h,text in bottom:
    box = FancyBboxPatch(
        (x,y), w,h,
        boxstyle='round,pad=0.012,rounding_size=0.012',
        linewidth=1.0, edgecolor='black', facecolor='white'
    )
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=BASE_FONT_SIZE)

# arrows from OOF
ox,oy,ow,oh,_ = steps[-1]
for x,y,w,h,_ in bottom:
    ax.add_patch(FancyArrowPatch(
        (ox+ow/2, oy), (x+w/2, y+h),
        arrowstyle='-|>', mutation_scale=12,
        linewidth=0.9, connectionstyle='arc3,rad=0.05'
    ))

save_pubfig(fig, 'Figure_1_Leakage_Safe_Workflow')

# ---- Original notebook code cell 7 ----

# Figure 2 — per-egg MAE distribution. NO title.

plot_df = mae_wide.set_index('sample')[MODELS_MAIN]

fig, ax = plt.subplots(figsize=(8.6,4.8))
bp = ax.boxplot(
    [plot_df[m].values for m in MODELS_MAIN],
    labels=MODELS_MAIN,
    showmeans=True,
    meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':4},
    medianprops={'linewidth':1.4}
)
ax.set_ylabel('Per-egg MAE (days)')
ax.set_xlabel('Model')
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', linewidth=0.4, alpha=0.35)

save_pubfig(fig, 'Figure_2_Per_Egg_MAE_Distribution')

# ---- Original notebook code cell 8 ----

# Figure 3 — observed vs predicted for SVR, PLSR, ANN.
# Panel letters upper-left; model name lower-right. NO titles.

fig, axes = plt.subplots(1,3,figsize=(12.4,4.1),sharex=True,sharey=True)

for ax, label, model_name in zip(axes, ['(a)','(b)','(c)'], TOP3):
    g = oof[oof['model']==model_name].copy()
    ax.scatter(
        g['storage_days'], g['y_pred'],
        s=12, alpha=0.35, edgecolors='none'
    )
    ax.plot([0,21],[0,21], linestyle='--', linewidth=1.0)
    ax.set_xlim(-0.5,21.5)
    ax.set_ylim(-3,24)
    ax.set_xlabel('Observed storage time (days)')
    ax.grid(linewidth=0.35, alpha=0.25)
    panel_label(ax,label)
    model_tag(ax,model_name)

axes[0].set_ylabel('Predicted storage time (days)')
fig.tight_layout()
save_pubfig(fig, 'Figure_3_Observed_vs_Predicted_Top3')

# ---- Original notebook code cell 9 ----

# Figure 4 — wavelength-order ablation. NO title.

summary_order = (
    order_egg
    .groupby(['model','order_condition'],as_index=False)
    .agg(
        mean_MAE=('MAE_days','mean'),
        SEM=('MAE_days',lambda x: float(np.std(x,ddof=1)/np.sqrt(len(x))))
    )
)

fig, ax = plt.subplots(figsize=(8.7,4.9))
x = np.arange(len(ORDER_CONDITIONS))

for model_name in DEEP_MODELS:
    g = (
        summary_order[summary_order['model']==model_name]
        .set_index('order_condition')
        .loc[ORDER_CONDITIONS]
    )
    ax.errorbar(
        x, g['mean_MAE'], yerr=1.96*g['SEM'],
        marker='o', linewidth=1.4, capsize=3,
        label=model_name
    )

ax.set_xticks(x)
ax.set_xticklabels(['Original','Reversed','Shuffled'])
ax.set_ylabel('Mean per-egg MAE (days)')
ax.set_xlabel('Wavelength-order condition')
ax.legend(frameon=False)
ax.grid(axis='y', linewidth=0.4, alpha=0.3)

save_pubfig(fig, 'Figure_4_Wavelength_Order_Ablation')

# ---- Original notebook code cell 10 ----

# Figure 5 — MAE and bias across storage day for top competitive models.
# Two panels; NO title.

fig, axes = plt.subplots(2,1,figsize=(9.4,7.0),sharex=True)

for model_name in TOP3:
    g = day[day['model']==model_name].sort_values('storage_days')
    axes[0].plot(g['storage_days'], g['MAE_days'], marker='o', markersize=3, linewidth=1.3, label=model_name)
    axes[1].plot(g['storage_days'], g['bias_days'], marker='o', markersize=3, linewidth=1.3, label=model_name)

axes[0].set_ylabel('MAE (days)')
axes[0].grid(linewidth=0.35, alpha=0.3)
axes[0].legend(frameon=False)
panel_label(axes[0], '(a)')

axes[1].axhline(0, linewidth=0.9, linestyle='--')
axes[1].set_ylabel('Bias (predicted − observed days)')
axes[1].set_xlabel('Storage day')
axes[1].grid(linewidth=0.35, alpha=0.3)
panel_label(axes[1], '(b)')

fig.tight_layout()
save_pubfig(fig, 'Figure_5_Error_and_Bias_Across_Storage_Time')

# ---- Original notebook code cell 11 ----

# Figure 6 — accuracy-latency reference plane. NO title.
# Log x-axis because recurrent models are orders of magnitude slower in this CPU benchmark.

lat1 = latency[latency['batch_size']==1].copy()
perf = operational[operational['model'].isin(MODELS_MAIN)][['model','MAE_days']]

pareto = perf.merge(
    lat1[['model','end_to_end_median_ms_per_spectrum']],
    on='model', how='inner'
)
assert len(pareto)==6

fig, ax = plt.subplots(figsize=(7.7,4.8))
ax.scatter(
    pareto['end_to_end_median_ms_per_spectrum'],
    pareto['MAE_days'],
    s=42
)
for _,r in pareto.iterrows():
    ax.annotate(
        r['model'],
        (r['end_to_end_median_ms_per_spectrum'],r['MAE_days']),
        xytext=(5,4), textcoords='offset points'
    )

ax.set_xscale('log')
ax.set_xlabel('Median end-to-end CPU latency (ms/spectrum, log scale)')
ax.set_ylabel('OOF MAE (days)')
ax.grid(which='both', linewidth=0.35, alpha=0.3)

save_pubfig(fig, 'Figure_6_Accuracy_Latency_Plane')
