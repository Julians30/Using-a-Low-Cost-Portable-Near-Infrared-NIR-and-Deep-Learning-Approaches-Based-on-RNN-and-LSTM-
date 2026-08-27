"""Public source fragment for NB05_WAVELENGTH_ORDER_ABLATION.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 7 ----

# Strict revalidation of NB04 configuration table and original OOF reference

expected_pairs = set(zip(df['sample'].astype(int), df['storage_days'].astype(int)))

assert set(selected_nb04['model']) == set(MODELS)
assert set(selected_nb04['outer_fold']) == set(range(1,6))
assert set(original_nb04['model']) == set(MODELS)
assert set(original_nb04['seed'].astype(int)) == set(FINAL_SEEDS)

for model_name in MODELS:
    for seed in FINAL_SEEDS:
        g = original_nb04[
            (original_nb04['model'] == model_name) &
            (original_nb04['seed'].astype(int) == seed)
        ]
        assert len(g) == 660
        assert set(zip(g['sample'].astype(int), g['storage_days'].astype(int))) == expected_pairs

# Every NB04 selected configuration must match its prediction metadata
merged_check = original_nb04.merge(
    selected_nb04[['outer_fold','model','selected_preprocessing','selected_epoch']],
    on=['outer_fold','model'],
    suffixes=('_pred','_sel'),
    validate='many_to_one'
)
assert (merged_check['selected_preprocessing_pred'] == merged_check['selected_preprocessing_sel']).all()
assert (merged_check['selected_epoch_pred'].astype(int) == merged_check['selected_epoch_sel'].astype(int)).all()

print('PASS — NB04 configurations and 7,920 original-order OOF predictions verified.')
display(selected_nb04.sort_values(['outer_fold','model']))

# ---- Original notebook code cell 8 ----

# Resumable checkpoints for ablated conditions
CP_METRICS = CHECKPOINT_DIR / 'NB05_ablation_outer_fold_seed_metrics.csv'
CP_PRED = CHECKPOINT_DIR / 'NB05_ablation_oof_predictions_seedwise.csv'

def load_cp(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()

def save_cp(df_, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df_.to_csv(tmp, index=False)
    os.replace(tmp, path)

ablation_metrics = load_cp(CP_METRICS)
ablation_pred = load_cp(CP_PRED)

print('Checkpoint rows:', {'metrics': len(ablation_metrics), 'predictions': len(ablation_pred)})

# ---- Original notebook code cell 9 ----

# Main NB05 ablation: only reversed + fixed shuffled conditions are newly trained
run_start = time.perf_counter()

for outer_fold in range(1, 6):
    print(f'\n================ OUTER FOLD {outer_fold}/5 ================')
    train_eggs, test_eggs = get_outer_train_test_eggs(outer_fold)

    train_mask = df['sample'].isin(train_eggs).to_numpy()
    test_mask = df['sample'].isin(test_eggs).to_numpy()

    X_train_raw = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test_raw = X_all[test_mask]
    y_test = y_all[test_mask]
    test_meta = df.loc[test_mask, ['sample','storage_days']].reset_index(drop=True)

    assert len(X_train_raw) == 528 and len(X_test_raw) == 132

    for model_name in MODELS:
        cfg = selected_nb04[
            (selected_nb04['outer_fold'] == outer_fold) &
            (selected_nb04['model'] == model_name)
        ]
        assert len(cfg) == 1
        prep_name = str(cfg.iloc[0]['selected_preprocessing'])
        selected_epoch = int(cfg.iloc[0]['selected_epoch'])

        # Fit exactly once on outer-training eggs, always in true wavelength order.
        pp = NeuralSpectralPreprocessor(prep_name, SG_WINDOW, SG_POLYORDER)
        X_train_pp = pp.fit_transform(X_train_raw)
        X_test_pp = pp.transform(X_test_raw)

        for condition in TRAIN_CONDITIONS:
            print(f'\n--- {model_name} | outer {outer_fold}/5 | {condition} | prep={prep_name} | epoch={selected_epoch} ---')

            Xtr_order = apply_order(X_train_pp, condition)
            Xte_order = apply_order(X_test_pp, condition)

            for seed in FINAL_SEEDS:
                already = False
                if not ablation_metrics.empty:
                    already = (
                        (ablation_metrics['outer_fold'].astype(int) == outer_fold) &
                        (ablation_metrics['model'] == model_name) &
                        (ablation_metrics['seed'].astype(int) == seed) &
                        (ablation_metrics['order_condition'] == condition)
                    ).any()

                if already:
                    print(f'  seed {seed}: checkpoint found — skipped')
                    continue

                tf.keras.backend.clear_session()
                gc.collect()
                set_all_seeds(seed)

                model = build_model(model_name, N_FEATURES)
                Xtr_m = shape_for_model(Xtr_order, model_name)
                Xte_m = shape_for_model(Xte_order, model_name)

                t0 = time.perf_counter()
                hist = model.fit(
                    Xtr_m, y_train,
                    epochs=selected_epoch,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                    shuffle=True
                )
                train_time_s = time.perf_counter() - t0

                # Predict once; outer test has never influenced training.
                t1 = time.perf_counter()
                pred = model.predict(Xte_m, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
                infer_total = time.perf_counter() - t1
                infer_ms_per_spectrum = 1000 * infer_total / len(pred)

                assert len(pred) == 132 and np.isfinite(pred).all()

                row = {
                    'order_condition': condition,
                    'outer_fold': outer_fold,
                    'model': model_name,
                    'seed': seed,
                    'selected_preprocessing': prep_name,
                    'selected_epoch': selected_epoch,
                    **metrics_dict(y_test, pred),
                    'train_time_s': float(train_time_s),
                    'inference_ms_per_spectrum': float(infer_ms_per_spectrum),
                    'parameter_count': int(model.count_params()),
                    'final_train_loss': float(hist.history['loss'][-1]),
                    'final_train_mae': float(hist.history['mae'][-1])
                }
                ablation_metrics = pd.concat(
                    [ablation_metrics, pd.DataFrame([row])],
                    ignore_index=True
                )

                pred_rows = test_meta.copy()
                pred_rows['outer_fold'] = outer_fold
                pred_rows['model'] = model_name
                pred_rows['seed'] = seed
                pred_rows['order_condition'] = condition
                pred_rows['y_pred'] = pred
                pred_rows['selected_preprocessing'] = prep_name
                pred_rows['selected_epoch'] = selected_epoch
                ablation_pred = pd.concat(
                    [ablation_pred, pred_rows],
                    ignore_index=True
                )

                save_cp(ablation_metrics, CP_METRICS)
                save_cp(ablation_pred, CP_PRED)

                print(
                    f"  seed {seed}: MAE={row['MAE_days']:.4f} | "
                    f"RMSE={row['RMSE_days']:.4f} | R2={row['R2']:.4f} | "
                    f"train={train_time_s:.1f}s"
                )

                del model, hist, Xtr_m, Xte_m
                tf.keras.backend.clear_session()
                gc.collect()

run_elapsed_s = time.perf_counter() - run_start
print(f'\nAblated-condition loop wall time this session: {run_elapsed_s/60:.1f} min')

# ---- Original notebook code cell 10 ----

# Completeness gate and integration with original NB04 predictions

ablation_metrics = load_cp(CP_METRICS)
ablation_pred = load_cp(CP_PRED)

assert len(ablation_metrics) == 2 * 5 * 4 * 3, f'Expected 120 ablation metric rows, got {len(ablation_metrics)}'
assert not ablation_metrics.duplicated(['order_condition','outer_fold','model','seed']).any()
assert len(ablation_pred) == 2 * 5 * 4 * 3 * 132, f'Expected 15,840 ablation predictions, got {len(ablation_pred)}'
assert not ablation_pred.duplicated(['order_condition','outer_fold','model','seed','sample','storage_days']).any()
assert np.isfinite(ablation_pred['y_pred']).all()

# Build original condition from the exact approved NB04 seed-wise OOF predictions.
orig = original_nb04[
    ['sample','storage_days','outer_fold','model','seed','y_pred',
     'selected_preprocessing','selected_epoch']
].copy()
orig['order_condition'] = 'original'

combined_pred = pd.concat(
    [orig, ablation_pred[orig.columns]],
    ignore_index=True
)
combined_pred['seed'] = combined_pred['seed'].astype(int)
combined_pred['outer_fold'] = combined_pred['outer_fold'].astype(int)

assert len(combined_pred) == 3 * 4 * 3 * 660, f'Expected 23,760 combined predictions, got {len(combined_pred)}'
assert not combined_pred.duplicated(['order_condition','model','seed','sample','storage_days']).any()

for condition in ORDER_CONDITIONS:
    for model_name in MODELS:
        for seed in FINAL_SEEDS:
            g = combined_pred[
                (combined_pred['order_condition'] == condition) &
                (combined_pred['model'] == model_name) &
                (combined_pred['seed'] == seed)
            ]
            assert len(g) == 660
            assert set(zip(g['sample'].astype(int), g['storage_days'].astype(int))) == expected_pairs

combined_pred.to_csv(RESULT_DIR / 'NB05_oof_predictions_seedwise_all_orders.csv', index=False)
ablation_metrics.to_csv(RESULT_DIR / 'NB05_new_ablation_outer_fold_seed_metrics.csv', index=False)

print('PASS — 23,760 OOF predictions across original/reversed/shuffled are complete.')

# ---- Original notebook code cell 11 ----

# Pooled, fold-wise, seed-mean, egg-level, and delta summaries

# Recompute all fold/seed metrics from predictions so original and ablations use identical code.
fold_seed_rows = []
for (condition, outer_fold, model_name, seed), g in combined_pred.groupby(
    ['order_condition','outer_fold','model','seed']
):
    fold_seed_rows.append({
        'order_condition': condition,
        'outer_fold': int(outer_fold),
        'model': model_name,
        'seed': int(seed),
        **metrics_dict(g['storage_days'], g['y_pred'])
    })
fold_seed_metrics = pd.DataFrame(fold_seed_rows)

pooled_seed_rows = []
for (condition, model_name, seed), g in combined_pred.groupby(['order_condition','model','seed']):
    pooled_seed_rows.append({
        'order_condition': condition,
        'model': model_name,
        'seed': int(seed),
        **metrics_dict(g['storage_days'], g['y_pred'])
    })
pooled_seed_metrics = pd.DataFrame(pooled_seed_rows)

seedmean_pred = (
    combined_pred
    .groupby(['order_condition','model','sample','storage_days','outer_fold'], as_index=False)
    .agg(
        y_pred=('y_pred','mean'),
        seed_sd_pred=('y_pred','std'),
        selected_preprocessing=('selected_preprocessing','first'),
        selected_epoch=('selected_epoch','first')
    )
)
assert len(seedmean_pred) == 3 * 4 * 660

pooled_seedmean_rows = []
egg_seedmean_rows = []
for (condition, model_name), g in seedmean_pred.groupby(['order_condition','model']):
    pooled_seedmean_rows.append({
        'order_condition': condition,
        'model': model_name,
        **metrics_dict(g['storage_days'], g['y_pred'])
    })
    for sample, eg in g.groupby('sample'):
        egg_seedmean_rows.append({
            'order_condition': condition,
            'model': model_name,
            'sample': int(sample),
            **metrics_dict(eg['storage_days'], eg['y_pred'])
        })

pooled_seedmean = pd.DataFrame(pooled_seedmean_rows)
egg_seedmean = pd.DataFrame(egg_seedmean_rows)

# Primary descriptive order-effect deltas relative to original, paired by model/fold/seed.
orig_fs = fold_seed_metrics[fold_seed_metrics['order_condition'] == 'original'][
    ['outer_fold','model','seed','MAE_days','RMSE_days','R2']
].rename(columns={
    'MAE_days':'MAE_original',
    'RMSE_days':'RMSE_original',
    'R2':'R2_original'
})
alt_fs = fold_seed_metrics[fold_seed_metrics['order_condition'] != 'original'].copy()
deltas = alt_fs.merge(orig_fs, on=['outer_fold','model','seed'], validate='many_to_one')
deltas['delta_MAE_days_vs_original'] = deltas['MAE_days'] - deltas['MAE_original']
deltas['delta_RMSE_days_vs_original'] = deltas['RMSE_days'] - deltas['RMSE_original']
deltas['delta_R2_vs_original'] = deltas['R2'] - deltas['R2_original']

delta_summary = (
    deltas.groupby(['order_condition','model'], as_index=False)
    .agg(
        mean_delta_MAE_days=('delta_MAE_days_vs_original','mean'),
        sd_delta_MAE_days=('delta_MAE_days_vs_original','std'),
        median_delta_MAE_days=('delta_MAE_days_vs_original','median'),
        mean_delta_RMSE_days=('delta_RMSE_days_vs_original','mean'),
        mean_delta_R2=('delta_R2_vs_original','mean'),
        n_paired_fold_seed=('delta_MAE_days_vs_original','size')
    )
)

fold_seed_metrics.to_csv(RESULT_DIR / 'NB05_outer_fold_seed_metrics_all_orders.csv', index=False)
pooled_seed_metrics.to_csv(RESULT_DIR / 'NB05_pooled_seedwise_metrics_all_orders.csv', index=False)
seedmean_pred.to_csv(RESULT_DIR / 'NB05_oof_predictions_seedmean_all_orders.csv', index=False)
pooled_seedmean.to_csv(RESULT_DIR / 'NB05_pooled_seedmean_metrics_all_orders.csv', index=False)
egg_seedmean.to_csv(RESULT_DIR / 'NB05_per_egg_seedmean_metrics_all_orders.csv', index=False)
deltas.to_csv(RESULT_DIR / 'NB05_paired_fold_seed_deltas_vs_original.csv', index=False)
delta_summary.to_csv(RESULT_DIR / 'NB05_order_effect_summary.csv', index=False)

print('Seed-mean pooled metrics by order condition:')
display(pooled_seedmean.sort_values(['model','MAE_days']))
print('\nDescriptive deltas vs original (formal inference deferred to NB06):')
display(delta_summary.sort_values(['model','order_condition']))
