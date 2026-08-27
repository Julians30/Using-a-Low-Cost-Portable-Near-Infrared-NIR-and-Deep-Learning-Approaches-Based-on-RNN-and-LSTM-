"""Public source fragment for NB03_CHEMOMETRIC_BASELINES.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 14 ----
# Diagnostic figures (publication figures will be rebuilt in NB08)
for model_name in ['DummyMean', 'PLSR', 'SVR']:
    m = oof[oof['model'] == model_name]
    plt.figure(figsize=(6, 6))
    plt.scatter(m['storage_days'], m['y_pred'], alpha=0.55, s=18)
    lo = min(m['storage_days'].min(), m['y_pred'].min())
    hi = max(m['storage_days'].max(), m['y_pred'].max())
    plt.plot([lo, hi], [lo, hi], '--', linewidth=1)
    plt.xlabel('Observed storage day')
    plt.ylabel('Predicted storage day')
    plt.title(f'NB03 OOF predicted vs observed — {model_name}')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'NB03_predicted_vs_observed_{model_name}.png', dpi=220)
    plt.close()

plt.figure(figsize=(7, 5))
for model_name in ['DummyMean', 'PLSR', 'SVR']:
    g = by_day[by_day['model'] == model_name]
    plt.plot(g['storage_days'], g['MAE_days'], marker='o', markersize=3, label=model_name)
plt.xlabel('Storage day')
plt.ylabel('MAE (days)')
plt.title('NB03 OOF absolute error by storage day')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB03_MAE_by_storage_day.png', dpi=220)
plt.close()

print('Diagnostic figures saved to:', FIG_DIR)

# ---- Original notebook code cell 15 ----
# Save prespecified search protocol and run summary
protocol = {
    'notebook': 'NB03_CHEMOMETRIC_BASELINES',
    'search_revision': SEARCH_REVISION,
    'notebook_filename': NOTEBOOK_FILENAME,
    'grid_expansion_stop_rule': GRID_EXPANSION_STOP_RULE,
    'dataset_sha256': dataset_sha,
    'outer_split_seed': int(split_manifest['outer_seed']),
    'outer_folds': int(split_manifest['outer_folds']),
    'inner_folds': int(split_manifest['inner_folds']),
    'grouping_unit': 'sample (egg)',
    'primary_selection_metric': 'MAE_days',
    'preprocessing_candidates': PREPROCESSING_CANDIDATES,
    'sg_window': SG_WINDOW,
    'sg_polyorder': SG_POLYORDER,
    'plsr_components': PLS_COMPONENTS,
    'svr_kernel': 'rbf',
    'svr_C': SVR_C,
    'svr_epsilon': SVR_EPSILON,
    'svr_gamma': SVR_GAMMA,
    'svr_standardization': 'StandardScaler fitted on training partition only',
    'msc_reference': 'mean spectrum fitted on training partition only',
    'formal_statistical_comparison_deferred_to': 'NB06_STATISTICAL_ROBUSTNESS',
    'hyperparameter_boundary_audit_file': 'NB03_hyperparameter_boundary_audit.csv',
    'hyperparameter_boundary_summary_file': 'NB03_hyperparameter_boundary_summary.csv',
    'runtime_seconds': elapsed_total
}
(RESULT_DIR / 'NB03_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

summary = {
    'status': 'COMPLETED',
    'search_revision': SEARCH_REVISION,
    'notebook_filename': NOTEBOOK_FILENAME,
    'grid_expansion_stop_rule': GRID_EXPANSION_STOP_RULE,
    'n_models': 3,
    'models': ['DummyMean', 'PLSR', 'SVR'],
    'oof_predictions_per_model': 660,
    'n_eggs': 30,
    'n_days': 22,
    'zero_outer_group_leakage': True,
    'selection_training_only': True,
    'hyperparameter_boundary_audit_file': 'NB03_hyperparameter_boundary_audit.csv',
    'hyperparameter_boundary_summary_file': 'NB03_hyperparameter_boundary_summary.csv',
    'runtime_seconds': elapsed_total
}
(RESULT_DIR / 'NB03_run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
