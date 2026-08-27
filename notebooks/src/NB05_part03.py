"""Public source fragment for NB05_WAVELENGTH_ORDER_ABLATION.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 12 ----

# Diagnostic figures — publication figures will be rebuilt in NB08

metric_plot = pooled_seedmean.copy()
condition_order = ['original','reversed','shuffled']

plt.figure(figsize=(8,5))
for model_name in MODELS:
    g = metric_plot[metric_plot['model'] == model_name].set_index('order_condition').loc[condition_order]
    plt.plot(condition_order, g['MAE_days'], marker='o', label=model_name)
plt.xlabel('Wavelength-order condition')
plt.ylabel('OOF MAE (days)')
plt.title('NB05 wavelength-order ablation — MAE')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB05_MAE_by_order_condition.png', dpi=220)
plt.close()

plt.figure(figsize=(8,5))
for model_name in MODELS:
    g = metric_plot[metric_plot['model'] == model_name].set_index('order_condition').loc[condition_order]
    plt.plot(condition_order, g['R2'], marker='o', label=model_name)
plt.xlabel('Wavelength-order condition')
plt.ylabel('OOF R²')
plt.title('NB05 wavelength-order ablation — R²')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB05_R2_by_order_condition.png', dpi=220)
plt.close()

# Delta MAE vs original
pivot = delta_summary.pivot(index='model', columns='order_condition', values='mean_delta_MAE_days')
pivot = pivot.reindex(MODELS)
ax = pivot.plot(kind='bar', figsize=(8,5))
ax.set_xlabel('Model')
ax.set_ylabel('Mean ΔMAE vs original (days)')
ax.set_title('NB05 order sensitivity — positive values indicate worse MAE')
plt.tight_layout()
plt.savefig(FIG_DIR / 'NB05_delta_MAE_vs_original.png', dpi=220)
plt.close()

print('Diagnostic figures saved to:', FIG_DIR)

# ---- Original notebook code cell 13 ----

# Freeze protocol and execution status

protocol = {
    'notebook': 'NB05_WAVELENGTH_ORDER_ABLATION',
    'notebook_filename': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'dataset_sha256': dataset_sha,
    'frozen_split_manifest_sha256': split_manifest_sha,
    'source_NB04_revision': EXPECTED_NB04_REVISION,
    'models': MODELS,
    'seeds': FINAL_SEEDS,
    'conditions': ORDER_CONDITIONS,
    'newly_trained_conditions': TRAIN_CONDITIONS,
    'original_condition_source': 'Exact approved NB04 seed-wise OOF predictions; no retraining',
    'shuffle_seed': SHUFFLE_SEED,
    'shuffle_target_independent': True,
    'preprocessing_rule': (
        'Use exact preprocessing selected by NB04 for each model × outer fold; '
        'fit on 24 outer-training eggs only; transform in physical wavelength order; '
        'apply order ablation only after preprocessing and StandardScaler.'
    ),
    'epoch_rule': 'Use exact NB04 selected epoch for each model × outer fold; no early stopping or retuning in NB05.',
    'outer_test_used_for_selection': False,
    'ablation_retuning': False,
    'recurrent_axis': '331 wavelengths, not storage time',
    'formal_order_effect_inference_deferred_to': 'NB06_STATISTICAL_ROBUSTNESS'
}
(RESULT_DIR / 'NB05_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

summary = {
    'status': 'COMPLETED',
    'run_revision': RUN_REVISION,
    'n_models': len(MODELS),
    'n_order_conditions': len(ORDER_CONDITIONS),
    'n_newly_trained_conditions': len(TRAIN_CONDITIONS),
    'n_outer_folds': 5,
    'n_seeds': len(FINAL_SEEDS),
    'new_ablation_final_fits_expected': 2 * 5 * 4 * 3,
    'combined_seedwise_oof_predictions': int(len(combined_pred)),
    'combined_seedmean_oof_predictions': int(len(seedmean_pred)),
    'n_eggs': int(df['sample'].nunique()),
    'n_days': int(df['storage_days'].nunique()),
    'no_inner_cv_repeated': True,
    'NB04_recipe_frozen': True,
    'zero_outer_group_leakage': True,
    'gpu_required': REQUIRE_GPU
}
(RESULT_DIR / 'NB05_run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

state.update({
    'status': 'COMPLETED',
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'combined_seedwise_oof_predictions': int(len(combined_pred))
})
STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')

execution_status = {
    'status': 'COMPLETED',
    'notebook': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'dataset_sha256': dataset_sha,
    'split_manifest_sha256': split_manifest_sha,
    'source_NB04_revision': EXPECTED_NB04_REVISION,
    'assertions': {
        'new_final_fits_120': len(ablation_metrics) == 120,
        'new_predictions_15840': len(ablation_pred) == 15840,
        'combined_predictions_23760': len(combined_pred) == 23760,
        'no_duplicate_combined_keys': not combined_pred.duplicated(
            ['order_condition','model','seed','sample','storage_days']
        ).any(),
        'all_predictions_finite': bool(np.isfinite(combined_pred['y_pred']).all())
    }
}
assert all(execution_status['assertions'].values())
(RESULT_DIR / 'EXECUTION_STATUS.json').write_text(
    json.dumps(execution_status, indent=2), encoding='utf-8'
)

print('NB05 status: COMPLETED')
