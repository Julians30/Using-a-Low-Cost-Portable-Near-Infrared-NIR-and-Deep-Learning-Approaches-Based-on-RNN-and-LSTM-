"""Public source fragment for NB07_PRACTICAL_APPLICABILITY.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 17 ----

# Benchmark controlado de latencia CPU.
rng = np.random.default_rng(LATENCY_SEED)

def benchmark_sklearn(model_name, batch_size, repeats):
    estimator = engineering_models[model_name]
    pp = engineering_preprocessors[model_name]

    indices = rng.integers(0, len(X), size=(repeats + LATENCY_WARMUP, batch_size))
    model_only_ms = []
    e2e_ms = []

    # Warm-up
    for idx in indices[:LATENCY_WARMUP]:
        Z = pp.transform(X[idx])
        _ = estimator.predict(Z)

    for idx in indices[LATENCY_WARMUP:]:
        Xb = X[idx]

        Z = pp.transform(Xb)
        t0 = time.perf_counter()
        _ = estimator.predict(Z)
        model_only_ms.append((time.perf_counter()-t0)*1000.0/batch_size)

        t0 = time.perf_counter()
        Z = pp.transform(Xb)
        _ = estimator.predict(Z)
        e2e_ms.append((time.perf_counter()-t0)*1000.0/batch_size)

    return {
        'model_only_median_ms_per_spectrum': float(np.median(model_only_ms)),
        'model_only_p95_ms_per_spectrum': float(np.quantile(model_only_ms,0.95)),
        'end_to_end_median_ms_per_spectrum': float(np.median(e2e_ms)),
        'end_to_end_p95_ms_per_spectrum': float(np.quantile(e2e_ms,0.95))
    }

def deep_input(model_name, Z):
    Z = Z.astype(np.float32, copy=False)
    if model_name == 'ANN':
        return Z
    return Z[...,None]

def benchmark_deep(model_name, batch_size, repeats):
    model = deep_arch_models[model_name]
    pp = SpectralPreprocessor(rep_cfg[model_name]['preprocessing']).fit(X)

    indices = rng.integers(0, len(X), size=(repeats + LATENCY_WARMUP, batch_size))
    model_only_ms = []
    e2e_ms = []

    with tf.device('/CPU:0'):
        for idx in indices[:LATENCY_WARMUP]:
            Z = pp.transform(X[idx])
            tensor = tf.convert_to_tensor(deep_input(model_name,Z))
            _ = model(tensor, training=False).numpy()

        for idx in indices[LATENCY_WARMUP:]:
            Xb = X[idx]

            Z = pp.transform(Xb)
            tensor = tf.convert_to_tensor(deep_input(model_name,Z))
            t0 = time.perf_counter()
            _ = model(tensor, training=False).numpy()
            model_only_ms.append((time.perf_counter()-t0)*1000.0/batch_size)

            t0 = time.perf_counter()
            Z = pp.transform(Xb)
            tensor = tf.convert_to_tensor(deep_input(model_name,Z))
            _ = model(tensor, training=False).numpy()
            e2e_ms.append((time.perf_counter()-t0)*1000.0/batch_size)

    return {
        'model_only_median_ms_per_spectrum': float(np.median(model_only_ms)),
        'model_only_p95_ms_per_spectrum': float(np.quantile(model_only_ms,0.95)),
        'end_to_end_median_ms_per_spectrum': float(np.median(e2e_ms)),
        'end_to_end_p95_ms_per_spectrum': float(np.quantile(e2e_ms,0.95))
    }

latency_rows = []

for model_name in MODELS_MAIN:
    for batch_size, repeats in [(1,LATENCY_SINGLE_REPEATS),(32,LATENCY_BATCH_REPEATS)]:
        if model_name in ['PLSR','SVR']:
            res = benchmark_sklearn(model_name,batch_size,repeats)
            benchmark_type = 'fitted final engineering model on all data; no performance estimate'
        else:
            res = benchmark_deep(model_name,batch_size,repeats)
            benchmark_type = 'frozen architecture forward-pass; weights not used for performance'

        latency_rows.append({
            'model': model_name,
            'device': 'CPU',
            'batch_size': batch_size,
            'repeats': repeats,
            'warmup_repeats': LATENCY_WARMUP,
            'benchmark_type': benchmark_type,
            **res
        })
        print(model_name, 'batch', batch_size, res)

latency = pd.DataFrame(latency_rows)
latency['end_to_end_throughput_spectra_per_second'] = (
    1000.0 / latency['end_to_end_median_ms_per_spectrum']
)
latency.to_csv(RESULT_DIR / 'NB07_CPU_inference_latency.csv', index=False)

display(latency)

# ---- Original notebook code cell 18 ----

# Entrenamiento como descriptor secundario: conservar tiempos de NB04 si están disponibles.
training_summary = pd.DataFrame()

if NB04_TRAIN_METRICS.exists():
    train_df = pd.read_csv(NB04_TRAIN_METRICS)

    time_candidates = [
        c for c in train_df.columns
        if 'train' in c.lower() and ('time' in c.lower() or 'second' in c.lower())
    ]
    model_candidates = [c for c in ['model','Model'] if c in train_df.columns]

    if time_candidates and model_candidates:
        tc = time_candidates[0]
        mc = model_candidates[0]
        training_summary = (
            train_df
            .groupby(mc, as_index=False)
            .agg(
                total_recorded_training_seconds=(tc,'sum'),
                mean_recorded_training_seconds=(tc,'mean'),
                median_recorded_training_seconds=(tc,'median'),
                n_recorded_runs=(tc,'size')
            )
            .rename(columns={mc:'model'})
        )
        training_summary['total_recorded_training_hours'] = (
            training_summary['total_recorded_training_seconds']/3600.0
        )
        training_summary.to_csv(RESULT_DIR / 'NB07_DL_training_time_secondary.csv', index=False)
        display(training_summary)
    else:
        print('NB04 training metrics existe, pero no se detectó automáticamente una columna de tiempo.')
        print(train_df.columns.tolist())
else:
    print('NB04_outer_fold_seed_metrics.csv no disponible; se omite tiempo de entrenamiento secundario.')

# ---- Original notebook code cell 19 ----

# Tabla integrada de aplicabilidad práctica.
single_latency = latency[latency['batch_size']==1][[
    'model','model_only_median_ms_per_spectrum','end_to_end_median_ms_per_spectrum',
    'end_to_end_throughput_spectra_per_second'
]]

complexity_small = engineering_complexity[[
    'model','complexity_measure','complexity_value','serialized_size_MB',
    'representative_preprocessing'
]]

practical = (
    operational[operational['model'].isin(MODELS_MAIN)]
    .merge(single_latency,on='model',how='left')
    .merge(complexity_small,on='model',how='left')
)

practical['NB06_statistical_interpretation'] = practical['model'].apply(
    lambda m: (
        'Competitive top group; no significant pairwise difference among SVR/PLSR/ANN after Holm'
        if m in ['SVR','PLSR','ANN']
        else 'Significantly higher MAE than SVR/PLSR/ANN in NB06 pairwise comparisons'
    )
)

practical['deployment_claim'] = (
    'Proof-of-concept computational applicability only; external/device validation required'
)

practical.to_csv(RESULT_DIR / 'NB07_integrated_practical_applicability_table.csv', index=False)
display(practical)

# ---- Original notebook code cell 20 ----

# Figuras diagnósticas para NB07. Versiones editoriales finales se prepararán en NB08.

# 1. Tolerancias operativas.
tol = operational[operational['model'].isin(MODELS_MAIN)].copy()
x = np.arange(len(tol))
width = 0.24
plt.figure(figsize=(10,5))
plt.bar(x-width, tol['within_1d_pct'], width=width, label='±1 day')
plt.bar(x, tol['within_2d_pct'], width=width, label='±2 days')
plt.bar(x+width, tol['within_3d_pct'], width=width, label='±3 days')
plt.xticks(x, tol['model'], rotation=25)
plt.ylabel('Predictions within tolerance (%)')
plt.xlabel('Model')
plt.title('NB07 — Operational tolerance of OOF predictions')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB07_operational_tolerance.png', dpi=220)
plt.close()

# 2. MAE por fase.
phase_plot = phase_metrics[phase_metrics['model'].isin(MODELS_MAIN)].copy()
plt.figure(figsize=(9,5))
for model_name in MODELS_MAIN:
    g = (
        phase_plot[phase_plot['model']==model_name]
        .set_index('storage_phase')
        .loc[phase_order]
    )
    plt.plot(phase_order, g['MAE_days'], marker='o', label=model_name)
plt.ylabel('MAE (days)')
plt.xlabel('Storage phase')
plt.title('NB07 — Error across storage phases')
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB07_MAE_by_storage_phase.png', dpi=220)
plt.close()

# 3. Bias por día.
plt.figure(figsize=(10,5))
for model_name in MODELS_MAIN:
    g = day_metrics[day_metrics['model']==model_name].sort_values('storage_days')
    plt.plot(g['storage_days'], g['bias_days'], marker='o', markersize=3, label=model_name)
plt.axhline(0, linewidth=1)
plt.xlabel('Storage day')
plt.ylabel('Bias (predicted − observed days)')
plt.title('NB07 — Prediction bias across storage time')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB07_bias_by_storage_day.png', dpi=220)
plt.close()

# 4. Pareto simple: MAE vs end-to-end latency batch=1.
pareto = practical.copy()
plt.figure(figsize=(8,5))
plt.scatter(pareto['end_to_end_median_ms_per_spectrum'], pareto['MAE_days'])
for _, r in pareto.iterrows():
    plt.annotate(r['model'], (r['end_to_end_median_ms_per_spectrum'],r['MAE_days']),
                 xytext=(4,4), textcoords='offset points')
plt.xlabel('Median end-to-end CPU latency (ms/spectrum)')
plt.ylabel('OOF MAE (days)')
plt.title('NB07 — Accuracy–latency reference plane')
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB07_accuracy_latency_plane.png', dpi=220)
plt.close()

print('Figuras NB07 guardadas:', FIG_DIR)

# ---- Original notebook code cell 21 ----

# Síntesis automática conservadora.
svr = practical[practical['model']=='SVR'].iloc[0]
pls = practical[practical['model']=='PLSR'].iloc[0]
ann = practical[practical['model']=='ANN'].iloc[0]

lines = []
lines.append('NB07 — PRACTICAL APPLICABILITY')
lines.append('')
lines.append(
    f"OOF point estimates: SVR MAE={svr['MAE_days']:.3f}, RMSE={svr['RMSE_days']:.3f}, R2={svr['R2']:.3f}; "
    f"PLSR MAE={pls['MAE_days']:.3f}, RMSE={pls['RMSE_days']:.3f}, R2={pls['R2']:.3f}; "
    f"ANN MAE={ann['MAE_days']:.3f}, RMSE={ann['RMSE_days']:.3f}, R2={ann['R2']:.3f}."
)
lines.append(
    'NB06: no significant pairwise differences were detected among SVR, PLSR and ANN after Holm correction.'
)
lines.append(
    'NB06: all nine comparisons between the competitive trio (SVR/PLSR/ANN) and recurrent models were significant.'
)
lines.append('')
lines.append(
    'Clipping to [0,21] is reported only as a secondary operational sensitivity analysis; primary OOF metrics remain unclipped.'
)
lines.append(
    'Latency values are CPU reference measurements from the current Colab environment and must not be interpreted as latency on SCiO/mobile/embedded hardware.'
)
lines.append(
    'PLSR/SVR full-data refits are used only for engineering descriptors (size/latency), never as generalization estimates.'
)
lines.append(
    'The study remains a proof of concept under one acquisition campaign; external validation is required before deployment claims.'
)

summary_text = '\n'.join(lines)
(RESULT_DIR / 'NB07_practical_summary.txt').write_text(summary_text, encoding='utf-8')
print(summary_text)

# ---- Original notebook code cell 22 ----

# Protocolo, estado y auditoría de cierre.
protocol = {
    'notebook': 'NB07_PRACTICAL_APPLICABILITY',
    'notebook_filename': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'generalization_source': 'frozen egg-disjoint OOF predictions from NB03/NB04; statistical interpretation from NB06',
    'new_generalization_training_performed': False,
    'primary_performance_predictions_clipped': False,
    'secondary_operational_clipping_range_days': [0,21],
    'storage_phases': {'early':[0,7], 'middle':[8,14], 'late':[15,21]},
    'engineering_refit_PLSR_SVR': 'all 660 spectra, only for latency/model size; no performance estimate',
    'deep_latency_benchmark': 'frozen NB04 architecture; CPU forward-pass; weights not used for performance',
    'latency_device': 'CPU',
    'latency_single_repeats': LATENCY_SINGLE_REPEATS,
    'latency_batch_repeats': LATENCY_BATCH_REPEATS,
    'latency_warmup': LATENCY_WARMUP,
    'external_validation_available': False,
    'deployment_claim_level': 'proof-of-concept only',
    'dataset_sha256': EXPECTED_DATASET_SHA256,
    'frozen_split_manifest_sha256': EXPECTED_SPLIT_MANIFEST_SHA256
}
(RESULT_DIR / 'NB07_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

assert len(oof) == 4620
assert len(operational) == 7
assert len(phase_metrics) == 21
assert len(day_metrics) == 154
assert len(clipping) == 7
assert len(engineering_complexity) == 6
assert len(latency) == 12
assert all_top3_nonsig
assert all_top_vs_recurrent_sig
assert np.isfinite(operational[['MAE_days','RMSE_days','R2']].to_numpy()).all()
assert np.isfinite(latency[['model_only_median_ms_per_spectrum',
                            'end_to_end_median_ms_per_spectrum']].to_numpy()).all()

run_summary = {
    'status': 'COMPLETED',
    'run_revision': RUN_REVISION,
    'n_models_descriptive': 7,
    'n_main_models': 6,
    'oof_operational_predictions': int(len(oof)),
    'storage_phase_rows': int(len(phase_metrics)),
    'storage_day_rows': int(len(day_metrics)),
    'latency_rows': int(len(latency)),
    'complexity_rows': int(len(engineering_complexity)),
    'top3_nonsignificant_after_Holm_confirmed': all_top3_nonsig,
    'top3_vs_recurrent_significant_after_Holm_confirmed': all_top_vs_recurrent_sig,
    'primary_metrics_unclipped': True,
    'external_validation': False
}
(RESULT_DIR / 'NB07_run_summary.json').write_text(json.dumps(run_summary,indent=2),encoding='utf-8')

execution_status = {
    'status': 'COMPLETED',
    'notebook': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'assertions': {
        'oof_7x660': len(oof)==4620,
        'operational_7_models': len(operational)==7,
        'phases_7x3': len(phase_metrics)==21,
        'days_7x22': len(day_metrics)==154,
        'engineering_6_models': len(engineering_complexity)==6,
        'latency_6x2_batches': len(latency)==12,
        'NB06_top3_nonsignificance_preserved': all_top3_nonsig,
        'NB06_top3_vs_recurrent_significance_preserved': all_top_vs_recurrent_sig,
        'no_new_generalization_estimate_from_full_data_fit': True
    }
}
assert all(execution_status['assertions'].values())
(RESULT_DIR / 'EXECUTION_STATUS.json').write_text(json.dumps(execution_status,indent=2),encoding='utf-8')

print('NB07 status: COMPLETED')
