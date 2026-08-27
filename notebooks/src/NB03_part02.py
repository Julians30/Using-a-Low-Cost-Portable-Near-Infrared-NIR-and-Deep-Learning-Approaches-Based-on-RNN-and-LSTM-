"""Public source fragment for NB03_CHEMOMETRIC_BASELINES.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 9 ----
# Core nested-CV search
inner_foldwise_rows = []
inner_summary_rows = []
selected_rows = []
outer_metric_rows = []
oof_rows = []

start_total = time.perf_counter()

for outer_fold in range(1, 6):
    print(f'\n===== OUTER FOLD {outer_fold}/5 =====')
    outer_train_eggs, outer_test_eggs = get_outer_train_test_eggs(outer_fold)
    inner = load_inner_assignment(outer_fold)

    train_mask = df['sample'].isin(outer_train_eggs).to_numpy()
    test_mask = df['sample'].isin(outer_test_eggs).to_numpy()

    X_outer_train = X_all[train_mask]
    y_outer_train = y_all[train_mask]
    X_outer_test = X_all[test_mask]
    y_outer_test = y_all[test_mask]
    test_meta = df.loc[test_mask, ['sample', 'storage_days']].reset_index(drop=True)

    # --- DummyMean negative control ---
    dummy_pred = np.full(len(y_outer_test), y_outer_train.mean(), dtype=float)
    dm = metrics_dict(y_outer_test, dummy_pred)
    outer_metric_rows.append({'outer_fold': outer_fold, 'model': 'DummyMean', **dm})
    for i in range(len(test_meta)):
        oof_rows.append({
            'sample': int(test_meta.loc[i, 'sample']),
            'storage_days': int(test_meta.loc[i, 'storage_days']),
            'outer_fold': outer_fold,
            'model': 'DummyMean',
            'y_pred': float(dummy_pred[i]),
            'selected_preprocessing': 'none',
            'selected_hyperparameters': '{}'
        })

    # -------- PLSR inner search --------
    pls_candidates = []
    for prep_name in PREPROCESSING_CANDIDATES:
        for n_comp in PLS_COMPONENTS:
            fold_maes = []
            for inner_fold in range(1, 5):
                val_eggs = set(inner.loc[inner['inner_fold'] == inner_fold, 'sample'])
                itr_eggs = outer_train_eggs - val_eggs
                assert len(itr_eggs) == 18 and len(val_eggs) == 6
                assert not (itr_eggs & val_eggs)

                itr_mask = df['sample'].isin(itr_eggs).to_numpy()
                iva_mask = df['sample'].isin(val_eggs).to_numpy()

                X_itr, y_itr = X_all[itr_mask], y_all[itr_mask]
                X_iva, y_iva = X_all[iva_mask], y_all[iva_mask]

                pp = SpectralPreprocessor(prep_name, SG_WINDOW, SG_POLYORDER)
                X_itr_p = pp.fit_transform(X_itr)
                X_iva_p = pp.transform(X_iva)

                model = PLSRegression(n_components=n_comp, scale=True, max_iter=1000)
                model.fit(X_itr_p, y_itr)
                pred = model.predict(X_iva_p).ravel()
                mae = mean_absolute_error(y_iva, pred)
                fold_maes.append(float(mae))

                inner_foldwise_rows.append({
                    'outer_fold': outer_fold, 'model': 'PLSR',
                    'preprocessing': prep_name, 'n_components': n_comp,
                    'C': np.nan, 'epsilon': np.nan, 'gamma': np.nan,
                    'inner_fold': inner_fold, 'MAE_days': float(mae)
                })

            rec = {
                'outer_fold': outer_fold, 'model': 'PLSR',
                'preprocessing': prep_name, 'n_components': n_comp,
                'C': np.nan, 'epsilon': np.nan, 'gamma': np.nan,
                'mean_inner_MAE_days': float(np.mean(fold_maes)),
                'sd_inner_MAE_days': float(np.std(fold_maes, ddof=1))
            }
            pls_candidates.append(rec)
            inner_summary_rows.append(rec.copy())

    pls_best = sorted(
        pls_candidates,
        key=lambda r: (r['mean_inner_MAE_days'],
                       PREPROCESSING_CANDIDATES.index(r['preprocessing']),
                       r['n_components'])
    )[0]
    print('PLSR selected:', pls_best)

    # Refit selected PLSR on complete outer train
    pp = SpectralPreprocessor(pls_best['preprocessing'], SG_WINDOW, SG_POLYORDER)
    Xtr_p = pp.fit_transform(X_outer_train)
    Xte_p = pp.transform(X_outer_test)
    pls = PLSRegression(n_components=int(pls_best['n_components']), scale=True, max_iter=1000)
    t0 = time.perf_counter()
    pls.fit(Xtr_p, y_outer_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = pls.predict(Xte_p).ravel()
    inference_total = time.perf_counter() - t0

    mm = metrics_dict(y_outer_test, pred)
    outer_metric_rows.append({
        'outer_fold': outer_fold, 'model': 'PLSR', **mm,
        'train_time_s': train_time,
        'inference_ms_per_spectrum': 1000*inference_total/len(pred)
    })
    selected_rows.append({
        **pls_best,
        'outer_train_eggs': 24, 'outer_test_eggs': 6
    })
    config_json = json.dumps({
        'preprocessing': pls_best['preprocessing'],
        'n_components': int(pls_best['n_components'])
    }, sort_keys=True)
    for i in range(len(test_meta)):
        oof_rows.append({
            'sample': int(test_meta.loc[i, 'sample']),
            'storage_days': int(test_meta.loc[i, 'storage_days']),
            'outer_fold': outer_fold,
            'model': 'PLSR',
            'y_pred': float(pred[i]),
            'selected_preprocessing': pls_best['preprocessing'],
            'selected_hyperparameters': config_json
        })

    # -------- SVR inner search --------
    svr_candidates = []
    for prep_name in PREPROCESSING_CANDIDATES:
        for C in SVR_C:
            for eps in SVR_EPSILON:
                for gamma in SVR_GAMMA:
                    fold_maes = []
                    for inner_fold in range(1, 5):
                        val_eggs = set(inner.loc[inner['inner_fold'] == inner_fold, 'sample'])
                        itr_eggs = outer_train_eggs - val_eggs

                        itr_mask = df['sample'].isin(itr_eggs).to_numpy()
                        iva_mask = df['sample'].isin(val_eggs).to_numpy()
                        X_itr, y_itr = X_all[itr_mask], y_all[itr_mask]
                        X_iva, y_iva = X_all[iva_mask], y_all[iva_mask]

                        pp = SpectralPreprocessor(prep_name, SG_WINDOW, SG_POLYORDER)
                        X_itr_p = pp.fit_transform(X_itr)
                        X_iva_p = pp.transform(X_iva)

                        scaler = StandardScaler()
                        X_itr_s = scaler.fit_transform(X_itr_p)
                        X_iva_s = scaler.transform(X_iva_p)

                        model = SVR(kernel='rbf', C=C, epsilon=eps, gamma=gamma)
                        model.fit(X_itr_s, y_itr)
                        pred = model.predict(X_iva_s)
                        mae = mean_absolute_error(y_iva, pred)
                        fold_maes.append(float(mae))

                        inner_foldwise_rows.append({
                            'outer_fold': outer_fold, 'model': 'SVR',
                            'preprocessing': prep_name, 'n_components': np.nan,
                            'C': C, 'epsilon': eps, 'gamma': gamma,
                            'inner_fold': inner_fold, 'MAE_days': float(mae)
                        })

                    rec = {
                        'outer_fold': outer_fold, 'model': 'SVR',
                        'preprocessing': prep_name, 'n_components': np.nan,
                        'C': C, 'epsilon': eps, 'gamma': gamma,
                        'mean_inner_MAE_days': float(np.mean(fold_maes)),
                        'sd_inner_MAE_days': float(np.std(fold_maes, ddof=1))
                    }
                    svr_candidates.append(rec)
                    inner_summary_rows.append(rec.copy())

    svr_best = sorted(
        svr_candidates,
        key=lambda r: (r['mean_inner_MAE_days'],
                       PREPROCESSING_CANDIDATES.index(r['preprocessing']),
                       r['C'], r['epsilon'], r['gamma'])
    )[0]
    print('SVR selected:', svr_best)

    pp = SpectralPreprocessor(svr_best['preprocessing'], SG_WINDOW, SG_POLYORDER)
    Xtr_p = pp.fit_transform(X_outer_train)
    Xte_p = pp.transform(X_outer_test)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_p)
    Xte_s = scaler.transform(Xte_p)

    svr = SVR(kernel='rbf',
              C=float(svr_best['C']),
              epsilon=float(svr_best['epsilon']),
              gamma=float(svr_best['gamma']))
    t0 = time.perf_counter()
    svr.fit(Xtr_s, y_outer_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = svr.predict(Xte_s)
    inference_total = time.perf_counter() - t0

    mm = metrics_dict(y_outer_test, pred)
    outer_metric_rows.append({
        'outer_fold': outer_fold, 'model': 'SVR', **mm,
        'train_time_s': train_time,
        'inference_ms_per_spectrum': 1000*inference_total/len(pred)
    })
    selected_rows.append({
        **svr_best,
        'outer_train_eggs': 24, 'outer_test_eggs': 6
    })
    config_json = json.dumps({
        'preprocessing': svr_best['preprocessing'],
        'C': float(svr_best['C']),
        'epsilon': float(svr_best['epsilon']),
        'gamma': float(svr_best['gamma'])
    }, sort_keys=True)
    for i in range(len(test_meta)):
        oof_rows.append({
            'sample': int(test_meta.loc[i, 'sample']),
            'storage_days': int(test_meta.loc[i, 'storage_days']),
            'outer_fold': outer_fold,
            'model': 'SVR',
            'y_pred': float(pred[i]),
            'selected_preprocessing': svr_best['preprocessing'],
            'selected_hyperparameters': config_json
        })

elapsed_total = time.perf_counter() - start_total
print(f'\nNested benchmark completed in {elapsed_total/60:.2f} minutes.')

# ---- Original notebook code cell 10 ----
# Save nested-search results and validate OOF coverage
inner_foldwise = pd.DataFrame(inner_foldwise_rows)
inner_summary = pd.DataFrame(inner_summary_rows)
selected = pd.DataFrame(selected_rows)
outer_metrics = pd.DataFrame(outer_metric_rows)
oof = pd.DataFrame(oof_rows).sort_values(['model', 'sample', 'storage_days']).reset_index(drop=True)

# Every model must predict every one of the 660 observations exactly once OOF
for model_name in ['DummyMean', 'PLSR', 'SVR']:
    m = oof[oof['model'] == model_name]
    assert len(m) == 660, f'{model_name}: expected 660 OOF predictions, got {len(m)}'
    assert not m.duplicated(['sample', 'storage_days']).any()
    assert set(zip(m['sample'], m['storage_days'])) == set(zip(df['sample'], df['storage_days']))

assert oof['y_pred'].notna().all()
assert np.isfinite(oof['y_pred']).all()

inner_foldwise.to_csv(RESULT_DIR / 'NB03_inner_search_foldwise.csv', index=False)
inner_summary.to_csv(RESULT_DIR / 'NB03_inner_search_summary.csv', index=False)
selected.to_csv(RESULT_DIR / 'NB03_selected_configurations.csv', index=False)
outer_metrics.to_csv(RESULT_DIR / 'NB03_outer_fold_metrics.csv', index=False)
oof.to_csv(RESULT_DIR / 'NB03_oof_predictions.csv', index=False)

print('PASS — complete OOF coverage for DummyMean, PLSR, and SVR.')
display(selected)

# ---- Original notebook code cell 11 ----
# Hyperparameter-boundary audit — required before accepting NB03
boundary_rows = []
for _, r in selected.iterrows():
    if r['model'] == 'PLSR':
        boundary_rows.append({
            'outer_fold': int(r['outer_fold']),
            'model': 'PLSR',
            'parameter': 'n_components',
            'selected_value': float(r['n_components']),
            'grid_min': float(min(PLS_COMPONENTS)),
            'grid_max': float(max(PLS_COMPONENTS)),
            'at_lower_boundary': bool(float(r['n_components']) == min(PLS_COMPONENTS)),
            'at_upper_boundary': bool(float(r['n_components']) == max(PLS_COMPONENTS)),
        })
    elif r['model'] == 'SVR':
        for parameter, grid in [('C', SVR_C), ('epsilon', SVR_EPSILON), ('gamma', SVR_GAMMA)]:
            value = float(r[parameter])
            boundary_rows.append({
                'outer_fold': int(r['outer_fold']),
                'model': 'SVR',
                'parameter': parameter,
                'selected_value': value,
                'grid_min': float(min(grid)),
                'grid_max': float(max(grid)),
                'at_lower_boundary': bool(value == min(grid)),
                'at_upper_boundary': bool(value == max(grid)),
            })

boundary_audit = pd.DataFrame(boundary_rows)
boundary_audit.to_csv(RESULT_DIR / 'NB03_hyperparameter_boundary_audit.csv', index=False)

boundary_summary = (
    boundary_audit
    .groupby(['model','parameter'], as_index=False)
    .agg(
        n_outer_folds=('outer_fold','nunique'),
        lower_boundary_hits=('at_lower_boundary','sum'),
        upper_boundary_hits=('at_upper_boundary','sum')
    )
)
boundary_summary.to_csv(RESULT_DIR / 'NB03_hyperparameter_boundary_summary.csv', index=False)

print('HYPERPARAMETER BOUNDARY AUDIT')
display(boundary_summary)

if ((boundary_summary['lower_boundary_hits'] == 5) | (boundary_summary['upper_boundary_hits'] == 5)).any():
    print('WARNING: At least one parameter selected the same search boundary in all five outer folds.')
    print('NB03 must be reviewed before NB04; do not widen the grid based on outer-test metrics.')
else:
    print('PASS: no parameter selected the same search boundary in all five outer folds.')

# ---- Original notebook code cell 12 ----
# Pooled OOF metrics and per-egg errors
pooled_rows = []
egg_rows = []

for model_name, m in oof.groupby('model'):
    pooled_rows.append({'model': model_name, **metrics_dict(m['storage_days'], m['y_pred'])})
    for sample, g in m.groupby('sample'):
        egg_rows.append({
            'model': model_name,
            'sample': int(sample),
            **metrics_dict(g['storage_days'], g['y_pred'])
        })

pooled_metrics = pd.DataFrame(pooled_rows).sort_values('MAE_days').reset_index(drop=True)
per_egg_metrics = pd.DataFrame(egg_rows).sort_values(['model', 'sample'])

pooled_metrics.to_csv(RESULT_DIR / 'NB03_pooled_oof_metrics.csv', index=False)
per_egg_metrics.to_csv(RESULT_DIR / 'NB03_per_egg_metrics.csv', index=False)

print('Pooled out-of-fold metrics — descriptive only; formal inference is deferred to NB06.')
display(pooled_metrics)

# ---- Original notebook code cell 13 ----
# Error by storage day
day_rows = []
for (model_name, day), g in oof.groupby(['model', 'storage_days']):
    d = metrics_dict(g['storage_days'], g['y_pred'])
    day_rows.append({'model': model_name, 'storage_days': int(day), 'n': len(g), **d})

by_day = pd.DataFrame(day_rows)
by_day.to_csv(RESULT_DIR / 'NB03_metrics_by_storage_day.csv', index=False)
display(by_day.head())
