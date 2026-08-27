"""Public source fragment for NB06_STATISTICAL_ROBUSTNESS.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 9 ----

orows = []
for (cond,m,seed,sample),g in nb05.groupby(['order_condition','model','seed','sample']):
    orows.append({'order_condition':cond,'model':m,'seed':int(seed),'sample':int(sample),
                  'MAE_days':mae(g.storage_days,g.y_pred)})
order_seed = pd.DataFrame(orows)
order_egg = order_seed.groupby(['order_condition','model','sample'],as_index=False).agg(
    MAE_days=('MAE_days','mean'),
    seed_SD_MAE_days=('MAE_days','std')
)
assert len(order_egg)==360
order_seed.to_csv(RESULT_DIR/'NB06_order_per_egg_MAE_seedwise.csv',index=False)
order_egg.to_csv(RESULT_DIR/'NB06_order_per_egg_MAE_primary.csv',index=False)

ofr=[]; opairs=[]
for mi,m in enumerate(DEEP_MODELS):
    pw = order_egg[order_egg.model==m].pivot(
        index='sample',columns='order_condition',values='MAE_days'
    )[ORDER_CONDITIONS]
    f = stats.friedmanchisquare(pw.original,pw.reversed,pw.shuffled)
    W = float(f.statistic/(len(pw)*2))
    ofr.append({'model':m,'n_eggs':len(pw),'friedman_chi2':float(f.statistic),
                'df':2,'p_value':float(f.pvalue),'kendall_W':W,
                'significant_0.05':bool(f.pvalue<ALPHA)})
    local=[]
    for pj,(ca,cb) in enumerate(itertools.combinations(ORDER_CONDITIONS,2)):
        a=pw[ca].to_numpy(); b=pw[cb].to_numpy()
        stat,p_raw,nz=wilcoxon_safe(a,b)
        dmean,lo,hi=paired_bootstrap_delta(a,b,BOOTSTRAP_REPS,BOOTSTRAP_SEED+2000+mi*10+pj)
        local.append({
            'model':m,'condition_A':ca,'condition_B':cb,
            'delta_MAE_A_minus_B_days':dmean,
            'delta95_CI_low':lo,'delta95_CI_high':hi,
            'wilcoxon_statistic':stat,'p_raw':p_raw,
            'rank_biserial_A_minus_B':rank_biserial(a-b)
        })
    local=pd.DataFrame(local)
    local['p_holm_within_model']=holm_adjust(local.p_raw.to_numpy())
    local['significant_holm_0.05']=local.p_holm_within_model<ALPHA
    local['CI_excludes_zero']=(local.delta95_CI_high<0)|(local.delta95_CI_low>0)
    opairs.append(local)

order_friedman=pd.DataFrame(ofr)
order_pairwise=pd.concat(opairs,ignore_index=True)
order_friedman.to_csv(RESULT_DIR/'NB06_order_friedman_by_model.csv',index=False)
order_pairwise.to_csv(RESULT_DIR/'NB06_order_pairwise_wilcoxon_holm.csv',index=False)
display(order_friedman)
display(order_pairwise)

# ---- Original notebook code cell 10 ----

d03 = pd.read_csv(NB03_POOLED)
d04 = pd.read_csv(NB04_POOLED)
d03['source']='NB03 deterministic OOF'
d04['source']='NB04 seed-mean OOF predictions (descriptive only)'
desc = pd.concat([d03,d04],ignore_index=True,sort=False)
order = ['SVR','PLSR','ANN','BiLSTM','LSTM','SimpleRNN','DummyMean']
desc['model']=pd.Categorical(desc.model,categories=order,ordered=True)
desc=desc.sort_values('model').reset_index(drop=True)
desc['model']=desc.model.astype(str)
desc.to_csv(RESULT_DIR/'NB06_unified_descriptive_metrics_MAE_RMSE_R2.csv',index=False)
display(desc[['model','MAE_days','RMSE_days','R2','bias_days',
              'within_1d_pct','within_2d_pct','within_3d_pct','source']])

seed_stability = deep_seed.groupby(['model','seed'],as_index=False).agg(
    mean_per_egg_MAE_days=('MAE_days','mean'),
    SD_across_eggs=('MAE_days','std')
)
seed_summary = seed_stability.groupby('model',as_index=False).agg(
    mean_seed_MAE=('mean_per_egg_MAE_days','mean'),
    SD_between_seed_MAE=('mean_per_egg_MAE_days','std'),
    min_seed_MAE=('mean_per_egg_MAE_days','min'),
    max_seed_MAE=('mean_per_egg_MAE_days','max')
)
seed_stability.to_csv(RESULT_DIR/'NB06_seed_stability_by_seed.csv',index=False)
seed_summary.to_csv(RESULT_DIR/'NB06_seed_stability_summary.csv',index=False)
display(seed_summary)

# ---- Original notebook code cell 11 ----

plt.figure(figsize=(9,5))
plt.boxplot([primary_wide[m].to_numpy() for m in PRIMARY_MODELS],labels=PRIMARY_MODELS,showmeans=True)
plt.ylabel('MAE por huevo (días)')
plt.xlabel('Modelo')
plt.title('NB06 — Distribución del MAE por huevo')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR/'NB06_per_egg_MAE_boxplot.png',dpi=220)
plt.close()

top3=pairwise[pairwise.top3_pair].copy()
labels=[f"{a} − {b}" for a,b in zip(top3.model_A,top3.model_B)]
x=np.arange(len(top3))
means=top3.delta_MAE_A_minus_B_days.to_numpy()
lower=means-top3.delta95_CI_low.to_numpy()
upper=top3.delta95_CI_high.to_numpy()-means
plt.figure(figsize=(8,4.5))
plt.errorbar(x,means,yerr=[lower,upper],fmt='o',capsize=5)
plt.axhline(0,linewidth=1)
plt.xticks(x,labels,rotation=20)
plt.ylabel('Δ MAE pareado (días)')
plt.title('NB06 — SVR, PLSR y ANN: diferencias pareadas por huevo')
plt.tight_layout()
plt.savefig(FIG_DIR/'NB06_top3_paired_MAE_differences.png',dpi=220)
plt.close()

omeans=order_egg.groupby(['model','order_condition'],as_index=False).MAE_days.mean()
plt.figure(figsize=(8,5))
for m in DEEP_MODELS:
    g=omeans[omeans.model==m].set_index('order_condition').loc[ORDER_CONDITIONS]
    plt.plot(ORDER_CONDITIONS,g.MAE_days,marker='o',label=m)
plt.ylabel('MAE medio por huevo (días)')
plt.xlabel('Condición del orden espectral')
plt.title('NB06 — Sensibilidad al orden espectral')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR/'NB06_order_ablation_per_egg_MAE.png',dpi=220)
plt.close()

print('Figuras diagnósticas guardadas.')

# ---- Original notebook code cell 12 ----

top3 = pairwise[pairwise.top3_pair]
lines = [
    'NB06 — STATISTICAL ROBUSTNESS',
    '',
    f'Friedman principal: chi2={fr.statistic:.4f}, df={k-1}, p={fr.pvalue:.6g}, Kendall W={kendall_w:.4f}.',
    f'Comparaciones significativas después de Holm: {int(pairwise.significant_holm_0.05.sum())}/{len(pairwise)}.',
    f'Entre SVR, PLSR y ANN, significativas después de Holm: {int(top3.significant_holm_0.05.sum())}/{len(top3)}.',
    '',
    'Ausencia de significancia NO demuestra equivalencia.',
    'No se realizó prueba formal de equivalencia porque no se preespecificó un margen práctico.',
    '',
    'Las semillas son repeticiones algorítmicas, no unidades biológicas independientes.',
    'La inferencia primaria usa 30 huevos independientes.'
]
for _,r in order_friedman.iterrows():
    lines.append(f"Orden {r.model}: Friedman p={r.p_value:.6g}, Kendall W={r.kendall_W:.4f}.")
summary_text='\n'.join(lines)
(RESULT_DIR/'NB06_statistical_summary.txt').write_text(summary_text,encoding='utf-8')
print(summary_text)

# ---- Original notebook code cell 13 ----

protocol = {
    'notebook':'NB06_STATISTICAL_ROBUSTNESS',
    'notebook_filename':NOTEBOOK_FILENAME,
    'run_revision':RUN_REVISION,
    'primary_independent_unit':'egg',
    'n_independent_eggs':30,
    'repeated_measurements_per_egg':22,
    'primary_inferential_metric':'per-egg MAE in days',
    'primary_models':PRIMARY_MODELS,
    'dummy_role':'secondary naive reference; excluded from primary Friedman',
    'deep_learning_seed_rule':'per-egg MAE per seed, then average across seeds before inference',
    'primary_global_test':'Friedman',
    'primary_global_effect_size':"Kendall's W",
    'posthoc_test':'paired Wilcoxon signed-rank',
    'multiplicity_control':'Holm',
    'paired_effect_size':'signed rank-biserial correlation',
    'bootstrap_repetitions':BOOTSTRAP_REPS,
    'bootstrap_seed':BOOTSTRAP_SEED,
    'bootstrap_cluster_unit':'egg',
    'alpha':ALPHA,
    'RMSE_R2_role':'secondary descriptive',
    'ANOVA_used':False,
    'Brier_used':False,
    'equivalence_test_used':False,
    'dataset_sha256':EXPECTED_DATASET_SHA256,
    'frozen_split_manifest_sha256':EXPECTED_SPLIT_MANIFEST_SHA256
}
(RESULT_DIR/'NB06_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

run_summary = {
    'status':'COMPLETED',
    'run_revision':RUN_REVISION,
    'n_primary_models':6,
    'n_independent_eggs':30,
    'primary_per_egg_rows':int(len(primary_long)),
    'primary_pairwise_tests':int(len(pairwise)),
    'order_per_egg_rows':int(len(order_egg)),
    'order_pairwise_tests':int(len(order_pairwise)),
    'bootstrap_repetitions':BOOTSTRAP_REPS,
    'primary_friedman_p':float(fr.pvalue),
    'primary_kendall_W':kendall_w,
    'n_primary_pairwise_significant_holm':int(pairwise.significant_holm_0.05.sum()),
    'n_top3_pairwise_significant_holm':int(pairwise[pairwise.top3_pair].significant_holm_0.05.sum())
}
(RESULT_DIR/'NB06_run_summary.json').write_text(json.dumps(run_summary,indent=2),encoding='utf-8')

execution_status = {
    'status':'COMPLETED',
    'notebook':NOTEBOOK_FILENAME,
    'run_revision':RUN_REVISION,
    'assertions':{
        'primary_180_rows':len(primary_long)==180,
        'primary_pairwise_15':len(pairwise)==15,
        'order_360_rows':len(order_egg)==360,
        'order_pairwise_12':len(order_pairwise)==12,
        'all_primary_finite':bool(np.isfinite(primary_long.MAE_days).all()),
        'all_order_finite':bool(np.isfinite(order_egg.MAE_days).all()),
        'seed_not_independent_unit':True
    }
}
assert all(execution_status['assertions'].values())
(RESULT_DIR/'EXECUTION_STATUS.json').write_text(json.dumps(execution_status,indent=2),encoding='utf-8')
print('NB06 status: COMPLETED')
