"""Public source fragment for NB04_DEEP_LEARNING_BENCHMARK.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 11 ----
# Consolidate checkpointed results and strict completeness gates
inner_foldwise = load_cp(CP_INNER)
inner_summary = load_cp(CP_SUMMARY)
selected_df = load_cp(CP_SELECTED)
final_metrics_df = load_cp(CP_FINAL_METRICS)
oof_seed_df = load_cp(CP_PRED)

# Strict expected dimensions
assert len(selected_df) == 5 * 4, f'Expected 20 selected configurations, got {len(selected_df)}'
assert not selected_df.duplicated(['outer_fold','model']).any()
assert len(final_metrics_df) == 5 * 4 * 3, f'Expected 60 final seed metrics, got {len(final_metrics_df)}'
assert not final_metrics_df.duplicated(['outer_fold','model','seed']).any()
assert len(oof_seed_df) == 5 * 4 * 3 * 132, f'Expected 7920 seed-wise OOF predictions, got {len(oof_seed_df)}'
assert oof_seed_df['y_pred'].notna().all() and np.isfinite(oof_seed_df['y_pred']).all()

# Every model × seed must provide exactly one OOF prediction for each of the 660 spectra.
expected_pairs = set(zip(df['sample'], df['storage_days']))
for model_name in MODELS:
    for seed in FINAL_SEEDS:
        m = oof_seed_df[(oof_seed_df['model'] == model_name) & (oof_seed_df['seed'] == seed)]
        assert len(m) == 660, f'{model_name} seed {seed}: expected 660 OOF predictions, got {len(m)}'
        assert not m.duplicated(['sample','storage_days']).any()
        assert set(zip(m['sample'], m['storage_days'])) == expected_pairs

# Build seed-mean predictions (primary descriptive DL prediction for later paired comparison).
oof_seedmean = (
    oof_seed_df
    .groupby(['sample','storage_days','outer_fold','model'], as_index=False)
    .agg(
        y_pred=('y_pred','mean'),
        seed_sd_pred=('y_pred','std'),
        selected_preprocessing=('selected_preprocessing','first'),
        selected_epoch=('selected_epoch','first')
    )
)
assert len(oof_seedmean) == 4 * 660
assert not oof_seedmean.duplicated(['model','sample','storage_days']).any()

# Save official NB04 result files outside checkpoint folder.
inner_foldwise.sort_values(['outer_fold','model','preprocessing','inner_fold']).to_csv(RESULT_DIR / 'NB04_inner_preprocessing_foldwise.csv', index=False)
inner_summary.sort_values(['outer_fold','model','mean_inner_MAE_days']).to_csv(RESULT_DIR / 'NB04_inner_preprocessing_summary.csv', index=False)
selected_df.sort_values(['outer_fold','model']).to_csv(RESULT_DIR / 'NB04_selected_configurations.csv', index=False)
final_metrics_df.sort_values(['outer_fold','model','seed']).to_csv(RESULT_DIR / 'NB04_outer_fold_seed_metrics.csv', index=False)
oof_seed_df.sort_values(['model','seed','sample','storage_days']).to_csv(RESULT_DIR / 'NB04_oof_predictions_seedwise.csv', index=False)
oof_seedmean.sort_values(['model','sample','storage_days']).to_csv(RESULT_DIR / 'NB04_oof_predictions_seedmean.csv', index=False)

print('PASS — complete 4 models × 3 seeds × 660 OOF predictions.')
display(selected_df.sort_values(['outer_fold','model']))

# ---- Original notebook code cell 12 ----
# Descriptive pooled metrics, egg-level metrics, day-level errors, seed stability, and epoch-boundary audit

# Seed-wise pooled metrics (660 OOF predictions per model/seed)
pooled_seed_rows = []
for (model_name, seed), g in oof_seed_df.groupby(['model','seed']):
    pooled_seed_rows.append({'model': model_name, 'seed': int(seed), **metrics_dict(g['storage_days'], g['y_pred'])})
pooled_seed_metrics = pd.DataFrame(pooled_seed_rows).sort_values(['model','seed'])

# Metrics using the mean prediction across the three predefined seeds
pooled_mean_rows, egg_rows, day_rows = [], [], []
for model_name, g in oof_seedmean.groupby('model'):
    pooled_mean_rows.append({'model': model_name, **metrics_dict(g['storage_days'], g['y_pred'])})
    for sample, eg in g.groupby('sample'):
        egg_rows.append({'model': model_name, 'sample': int(sample), **metrics_dict(eg['storage_days'], eg['y_pred'])})
    for day, dg in g.groupby('storage_days'):
        day_rows.append({'model': model_name, 'storage_days': int(day), 'n': len(dg), **metrics_dict(dg['storage_days'], dg['y_pred'])})

pooled_seedmean_metrics = pd.DataFrame(pooled_mean_rows).sort_values('MAE_days').reset_index(drop=True)
per_egg_seedmean_metrics = pd.DataFrame(egg_rows).sort_values(['model','sample'])
by_day_seedmean = pd.DataFrame(day_rows).sort_values(['model','storage_days'])

# Seed stability from pooled MAE/RMSE.
seed_stability = (
    pooled_seed_metrics.groupby('model', as_index=False)
    .agg(
        MAE_mean_across_seeds=('MAE_days','mean'),
        MAE_sd_across_seeds=('MAE_days','std'),
        MAE_min=('MAE_days','min'),
        MAE_max=('MAE_days','max'),
        RMSE_mean_across_seeds=('RMSE_days','mean'),
        RMSE_sd_across_seeds=('RMSE_days','std')
    )
)

# Audit whether selected early-stopping epochs hit MAX_EPOCHS.
epoch_audit = (
    inner_foldwise.assign(hit_max_epoch=inner_foldwise['best_epoch'].astype(int) >= MAX_EPOCHS)
    .groupby(['outer_fold','model','preprocessing'], as_index=False)
    .agg(
        n_inner_folds=('inner_fold','nunique'),
        max_epoch_hits=('hit_max_epoch','sum'),
        median_best_epoch=('best_epoch','median'),
        max_best_epoch=('best_epoch','max')
    )
)

pooled_seed_metrics.to_csv(RESULT_DIR / 'NB04_pooled_seedwise_metrics.csv', index=False)
pooled_seedmean_metrics.to_csv(RESULT_DIR / 'NB04_pooled_seedmean_metrics.csv', index=False)
per_egg_seedmean_metrics.to_csv(RESULT_DIR / 'NB04_per_egg_seedmean_metrics.csv', index=False)
by_day_seedmean.to_csv(RESULT_DIR / 'NB04_metrics_by_storage_day_seedmean.csv', index=False)
seed_stability.to_csv(RESULT_DIR / 'NB04_seed_stability.csv', index=False)
epoch_audit.to_csv(RESULT_DIR / 'NB04_epoch_boundary_audit.csv', index=False)

print('Seed-mean OOF metrics — descriptive only; formal paired inference is deferred to NB06.')
display(pooled_seedmean_metrics)
print('\nSeed stability:')
display(seed_stability)

# ---- Original notebook code cell 13 ----
# Diagnostic figures — final publication graphics will be rebuilt in NB08 from frozen CSVs
for model_name in MODELS:
    g = oof_seedmean[oof_seedmean['model'] == model_name]
    plt.figure(figsize=(6, 6))
    plt.scatter(g['storage_days'], g['y_pred'], alpha=0.55, s=18)
    lo = min(g['storage_days'].min(), g['y_pred'].min())
    hi = max(g['storage_days'].max(), g['y_pred'].max())
    plt.plot([lo, hi], [lo, hi], '--', linewidth=1)
    plt.xlabel('Observed storage day')
    plt.ylabel('Predicted storage day (mean across 3 seeds)')
    plt.title(f'NB04 OOF predicted vs observed — {model_name}')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'NB04_predicted_vs_observed_{model_name}.png', dpi=220)
    plt.close()

plt.figure(figsize=(8, 5))
for model_name in MODELS:
    g = by_day_seedmean[by_day_seedmean['model'] == model_name]
    plt.plot(g['storage_days'], g['MAE_days'], marker='o', markersize=3, label=model_name)
plt.xlabel('Storage day')
plt.ylabel('MAE (days)')
plt.title('NB04 seed-mean OOF error by storage day')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB04_MAE_by_storage_day.png', dpi=220)
plt.close()

plt.figure(figsize=(7, 5))
for model_name in MODELS:
    g = pooled_seed_metrics[pooled_seed_metrics['model'] == model_name].sort_values('seed')
    plt.plot(g['seed'].astype(str), g['MAE_days'], marker='o', label=model_name)
plt.xlabel('Training seed')
plt.ylabel('Pooled OOF MAE (days)')
plt.title('NB04 seed stability')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB04_seed_stability_MAE.png', dpi=220)
plt.close()

print('Diagnostic figures saved to:', FIG_DIR)

# ---- Original notebook code cell 14 ----
# Freeze protocol, run summary, and execution status

model_specs = {
    'ANN': {'hidden_units': ANN_HIDDEN, 'dropout': DROPOUT},
    'SimpleRNN': {'recurrent_units': RECURRENT_UNITS, 'dense_units': DENSE_AFTER_RECURRENT, 'dropout': DROPOUT},
    'LSTM': {'recurrent_units': RECURRENT_UNITS, 'dense_units': DENSE_AFTER_RECURRENT, 'dropout': DROPOUT},
    'BiLSTM': {'recurrent_units_per_direction': RECURRENT_UNITS, 'dense_units': DENSE_AFTER_RECURRENT, 'dropout': DROPOUT},
}

protocol = {
    'notebook': 'NB04_DEEP_LEARNING_BENCHMARK',
    'notebook_filename': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'dataset_sha256': dataset_sha,
    'frozen_split_manifest_sha256': split_manifest_sha,
    'outer_split_seed': int(split_manifest['outer_seed']),
    'outer_folds': int(split_manifest['outer_folds']),
    'inner_folds': int(split_manifest['inner_folds']),
    'grouping_unit': 'sample (egg)',
    'models': MODELS,
    'preprocessing_candidates': PREPROCESSING_CANDIDATES,
    'preprocessing_selection': '4-fold frozen inner egg-disjoint CV; mean inner MAE',
    'epoch_selection': 'median best epoch across 4 inner folds for selected preprocessing',
    'architecture_hyperparameters': 'fixed a priori to reduce inner-CV over-search with n=30 eggs',
    'model_specs': model_specs,
    'optimizer': 'Adam',
    'learning_rate': LEARNING_RATE,
    'loss': LOSS,
    'batch_size': BATCH_SIZE,
    'max_epochs': MAX_EPOCHS,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
    'sg_window': SG_WINDOW,
    'sg_polyorder': SG_POLYORDER,
    'scaling': 'StandardScaler fitted after base preprocessing using training partition only',
    'msc_reference': 'mean spectrum fitted on training partition only',
    'final_seeds': FINAL_SEEDS,
    'recurrent_sequence_axis': '331 ordered wavelengths, 740→1070 nm; NOT storage time',
    'outer_test_used_for_selection': False,
    'formal_statistical_comparison_deferred_to': 'NB06_STATISTICAL_ROBUSTNESS',
    'wavelength_order_ablation_deferred_to': 'NB05_WAVELENGTH_ORDER_ABLATION',
    'resume_checkpointing': True
}
(RESULT_DIR / 'NB04_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

summary = {
    'status': 'COMPLETED',
    'run_revision': RUN_REVISION,
    'models': MODELS,
    'n_models': len(MODELS),
    'final_seeds': FINAL_SEEDS,
    'n_outer_folds': 5,
    'n_inner_folds': 4,
    'oof_predictions_per_model_per_seed': 660,
    'total_seedwise_oof_predictions': int(len(oof_seed_df)),
    'total_seedmean_oof_predictions': int(len(oof_seedmean)),
    'n_eggs': 30,
    'n_days': 22,
    'zero_outer_group_leakage': True,
    'selection_training_only': True,
    'architecture_capacity_fixed_a_priori': True,
    'recurrent_axis_is_wavelength_not_time': True,
    'gpu_required': REQUIRE_GPU,
}
(RESULT_DIR / 'NB04_run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

state.update({
    'status': 'COMPLETED',
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'selected_configurations': int(len(selected_df)),
    'final_seed_runs': int(len(final_metrics_df)),
    'seedwise_predictions': int(len(oof_seed_df))
})
STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')

execution_status = {
    'status': 'COMPLETED',
    'notebook': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'dataset_sha256': dataset_sha,
    'split_manifest_sha256': split_manifest_sha,
    'assertions_passed': [
        'dataset hash matches frozen value',
        'all NB02 split hashes match manifest',
        'zero outer egg overlap',
        'zero inner egg overlap',
        'no outer-test egg in inner selection',
        '20 selected architecture/fold configurations',
        '60 final outer model-seed runs',
        '7920 finite seed-wise OOF predictions',
        '660 OOF predictions per model per seed'
    ],
    'completed_at_utc': state['completed_at_utc']
}
(RESULT_DIR / 'EXECUTION_STATUS.json').write_text(json.dumps(execution_status, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
