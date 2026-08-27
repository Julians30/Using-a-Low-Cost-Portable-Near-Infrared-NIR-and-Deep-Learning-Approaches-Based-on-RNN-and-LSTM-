"""Public source fragment for NB04_DEEP_LEARNING_BENCHMARK.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 9 ----
# Checkpoint helpers

CP_INNER = CHECKPOINT_DIR / 'NB04_inner_preprocessing_foldwise.csv'
CP_SUMMARY = CHECKPOINT_DIR / 'NB04_inner_preprocessing_summary.csv'
CP_SELECTED = CHECKPOINT_DIR / 'NB04_selected_configurations.csv'
CP_FINAL_METRICS = CHECKPOINT_DIR / 'NB04_outer_fold_seed_metrics.csv'
CP_PRED = CHECKPOINT_DIR / 'NB04_oof_predictions_seedwise.csv'


def load_cp(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def save_cp(df_, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df_.to_csv(tmp, index=False)
    os.replace(tmp, path)

inner_foldwise = load_cp(CP_INNER)
inner_summary = load_cp(CP_SUMMARY)
selected_df = load_cp(CP_SELECTED)
final_metrics_df = load_cp(CP_FINAL_METRICS)
oof_seed_df = load_cp(CP_PRED)

print('Checkpoint rows loaded:', {
    'inner_foldwise': len(inner_foldwise),
    'inner_summary': len(inner_summary),
    'selected': len(selected_df),
    'final_metrics': len(final_metrics_df),
    'predictions': len(oof_seed_df)
})

# ---- Original notebook code cell 10 ----
# Main NB04 benchmark — resumable
run_start = time.perf_counter()

for outer_fold in range(1, 6):
    print(f'\n================ OUTER FOLD {outer_fold}/5 ================')
    outer_train_eggs, outer_test_eggs = get_outer_train_test_eggs(outer_fold)
    inner = load_inner_assignment(outer_fold)

    outer_train_mask = df['sample'].isin(outer_train_eggs).to_numpy()
    outer_test_mask = df['sample'].isin(outer_test_eggs).to_numpy()
    X_outer_train, y_outer_train = X_all[outer_train_mask], y_all[outer_train_mask]
    X_outer_test, y_outer_test = X_all[outer_test_mask], y_all[outer_test_mask]
    test_meta = df.loc[outer_test_mask, ['sample', 'storage_days']].reset_index(drop=True)

    assert len(X_outer_train) == 528 and len(X_outer_test) == 132

    for model_idx, model_name in enumerate(MODELS):
        print(f'\n--- {model_name} | outer {outer_fold}/5 ---')

        # -------- INNER SELECTION: preprocessing + epoch --------
        existing_sel = pd.DataFrame()
        if not selected_df.empty:
            existing_sel = selected_df[
                (selected_df['outer_fold'] == outer_fold) &
                (selected_df['model'] == model_name)
            ]

        if len(existing_sel) == 1:
            sel = existing_sel.iloc[0].to_dict()
            selected_prep = str(sel['selected_preprocessing'])
            selected_epoch = int(sel['selected_epoch'])
            print(f'Checkpoint selection: preprocessing={selected_prep}, epoch={selected_epoch}')
        else:
            # If a previous session stopped mid-selection, restart this architecture/fold cleanly.
            if not inner_foldwise.empty:
                inner_foldwise = inner_foldwise[~((inner_foldwise['outer_fold'] == outer_fold) & (inner_foldwise['model'] == model_name))].reset_index(drop=True)
                save_cp(inner_foldwise, CP_INNER)
            if not inner_summary.empty:
                inner_summary = inner_summary[~((inner_summary['outer_fold'] == outer_fold) & (inner_summary['model'] == model_name))].reset_index(drop=True)
                save_cp(inner_summary, CP_SUMMARY)
            candidate_rows = []
            for prep_idx, prep_name in enumerate(PREPROCESSING_CANDIDATES):
                fold_maes, fold_epochs = [], []
                for inner_fold in range(1, 5):
                    val_eggs = set(inner.loc[inner['inner_fold'] == inner_fold, 'sample'])
                    itr_eggs = outer_train_eggs - val_eggs
                    assert len(itr_eggs) == 18 and len(val_eggs) == 6
                    assert not (itr_eggs & val_eggs)
                    assert not (outer_test_eggs & (itr_eggs | val_eggs))

                    itr_mask = df['sample'].isin(itr_eggs).to_numpy()
                    iva_mask = df['sample'].isin(val_eggs).to_numpy()
                    X_itr, y_itr = X_all[itr_mask], y_all[itr_mask]
                    X_iva, y_iva = X_all[iva_mask], y_all[iva_mask]

                    pp = NeuralSpectralPreprocessor(prep_name, SG_WINDOW, SG_POLYORDER)
                    X_itr_p = pp.fit_transform(X_itr)
                    X_iva_p = pp.transform(X_iva)
                    X_itr_m = shape_for_model(X_itr_p, model_name)
                    X_iva_m = shape_for_model(X_iva_p, model_name)

                    # Same initialization seed across preprocessing candidates within each fold.
                    inner_seed = 50000 + outer_fold*100 + model_idx*10 + inner_fold
                    tf.keras.backend.clear_session()
                    set_all_seeds(inner_seed)
                    model = build_model(model_name, len(spectral_cols))
                    cb = keras.callbacks.EarlyStopping(
                        monitor='val_mae',
                        mode='min',
                        patience=EARLY_STOPPING_PATIENCE,
                        min_delta=EARLY_STOPPING_MIN_DELTA,
                        restore_best_weights=True,
                        verbose=0
                    )
                    t0 = time.perf_counter()
                    hist = model.fit(
                        X_itr_m, y_itr,
                        validation_data=(X_iva_m, y_iva),
                        epochs=MAX_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[cb],
                        verbose=0
                    )
                    fit_s = time.perf_counter() - t0
                    val_mae_hist = np.asarray(hist.history['val_mae'], dtype=float)
                    best_epoch = int(getattr(cb, 'best_epoch', int(np.argmin(val_mae_hist))) + 1)
                    pred = model.predict(X_iva_m, batch_size=128, verbose=0).ravel()
                    val_mae = float(mean_absolute_error(y_iva, pred))
                    fold_maes.append(val_mae)
                    fold_epochs.append(best_epoch)

                    row = {
                        'outer_fold': outer_fold,
                        'model': model_name,
                        'preprocessing': prep_name,
                        'inner_fold': inner_fold,
                        'inner_seed': inner_seed,
                        'n_train_eggs': 18,
                        'n_val_eggs': 6,
                        'n_train_spectra': int(len(y_itr)),
                        'n_val_spectra': int(len(y_iva)),
                        'best_epoch': best_epoch,
                        'epochs_ran': int(len(hist.history['loss'])),
                        'MAE_days': val_mae,
                        'fit_time_s': float(fit_s),
                        'hit_max_epoch': bool(best_epoch >= MAX_EPOCHS)
                    }
                    inner_foldwise = pd.concat([inner_foldwise, pd.DataFrame([row])], ignore_index=True)
                    save_cp(inner_foldwise, CP_INNER)

                    del model, X_itr_m, X_iva_m, X_itr_p, X_iva_p, pp
                    tf.keras.backend.clear_session()
                    gc.collect()

                cand = {
                    'outer_fold': outer_fold,
                    'model': model_name,
                    'preprocessing': prep_name,
                    'mean_inner_MAE_days': float(np.mean(fold_maes)),
                    'sd_inner_MAE_days': float(np.std(fold_maes, ddof=1)),
                    'median_best_epoch': int(np.rint(np.median(fold_epochs))),
                    'min_best_epoch': int(np.min(fold_epochs)),
                    'max_best_epoch': int(np.max(fold_epochs))
                }
                candidate_rows.append(cand)
                inner_summary = pd.concat([inner_summary, pd.DataFrame([cand])], ignore_index=True)
                save_cp(inner_summary, CP_SUMMARY)
                print(
                    f"  prep={prep_name:9s} | inner MAE={cand['mean_inner_MAE_days']:.4f} "
                    f"| median epoch={cand['median_best_epoch']}"
                )

            best = sorted(
                candidate_rows,
                key=lambda r: (
                    r['mean_inner_MAE_days'],
                    PREPROCESSING_CANDIDATES.index(r['preprocessing']),
                    r['median_best_epoch']
                )
            )[0]
            selected_prep = best['preprocessing']
            selected_epoch = int(np.clip(best['median_best_epoch'], 1, MAX_EPOCHS))
            sel_row = {
                'outer_fold': outer_fold,
                'model': model_name,
                'selected_preprocessing': selected_prep,
                'selected_epoch': selected_epoch,
                'mean_inner_MAE_days': best['mean_inner_MAE_days'],
                'sd_inner_MAE_days': best['sd_inner_MAE_days'],
                'outer_train_eggs': 24,
                'outer_test_eggs': 6,
                'architecture_capacity_fixed_a_priori': True
            }
            selected_df = pd.concat([selected_df, pd.DataFrame([sel_row])], ignore_index=True)
            save_cp(selected_df, CP_SELECTED)
            print(f"SELECTED: preprocessing={selected_prep}, epoch={selected_epoch}, inner MAE={best['mean_inner_MAE_days']:.4f}")

        # -------- FINAL OUTER REFIT: three predefined seeds --------
        pp_outer = NeuralSpectralPreprocessor(selected_prep, SG_WINDOW, SG_POLYORDER)
        Xtr_p = pp_outer.fit_transform(X_outer_train)
        Xte_p = pp_outer.transform(X_outer_test)
        Xtr_m = shape_for_model(Xtr_p, model_name)
        Xte_m = shape_for_model(Xte_p, model_name)

        for seed in FINAL_SEEDS:
            existing_pred = pd.DataFrame()
            if not oof_seed_df.empty:
                existing_pred = oof_seed_df[
                    (oof_seed_df['outer_fold'] == outer_fold) &
                    (oof_seed_df['model'] == model_name) &
                    (oof_seed_df['seed'] == seed)
                ]
            if len(existing_pred) == 132:
                print(f'  seed {seed}: checkpoint complete — skipped')
                continue
            else:
                # Remove incomplete predictions and any orphan metric row before rerun.
                if len(existing_pred) > 0:
                    oof_seed_df = oof_seed_df.drop(existing_pred.index).reset_index(drop=True)
                if not final_metrics_df.empty:
                    badm = final_metrics_df[
                        (final_metrics_df['outer_fold'] == outer_fold) &
                        (final_metrics_df['model'] == model_name) &
                        (final_metrics_df['seed'] == seed)
                    ].index
                    if len(badm) > 0:
                        final_metrics_df = final_metrics_df.drop(badm).reset_index(drop=True)

            tf.keras.backend.clear_session()
            set_all_seeds(seed)
            model = build_model(model_name, len(spectral_cols))
            t0 = time.perf_counter()
            hist = model.fit(
                Xtr_m, y_outer_train,
                epochs=selected_epoch,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0
            )
            train_s = time.perf_counter() - t0

            # Warm-up then timed inference.
            _ = model.predict(Xte_m[:min(16, len(Xte_m))], batch_size=128, verbose=0)
            t0 = time.perf_counter()
            pred = model.predict(Xte_m, batch_size=128, verbose=0).ravel()
            infer_s = time.perf_counter() - t0
            assert len(pred) == 132 and np.isfinite(pred).all()

            mm = metrics_dict(y_outer_test, pred)
            metric_row = {
                'outer_fold': outer_fold,
                'model': model_name,
                'seed': seed,
                'selected_preprocessing': selected_prep,
                'selected_epoch': selected_epoch,
                **mm,
                'train_time_s': float(train_s),
                'inference_ms_per_spectrum': float(1000*infer_s/len(pred)),
                'parameter_count': int(model.count_params()),
                'final_train_loss': float(hist.history['loss'][-1]),
                'final_train_mae': float(hist.history['mae'][-1])
            }
            final_metrics_df = pd.concat([final_metrics_df, pd.DataFrame([metric_row])], ignore_index=True)

            config_json = json.dumps({
                'preprocessing': selected_prep,
                'epoch': selected_epoch,
                'architecture_fixed': True,
                'units': RECURRENT_UNITS if model_name != 'ANN' else ANN_HIDDEN,
                'dropout': DROPOUT,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'loss': LOSS
            }, sort_keys=True)
            rows = []
            for i in range(len(test_meta)):
                rows.append({
                    'sample': int(test_meta.loc[i, 'sample']),
                    'storage_days': int(test_meta.loc[i, 'storage_days']),
                    'outer_fold': outer_fold,
                    'model': model_name,
                    'seed': seed,
                    'y_pred': float(pred[i]),
                    'selected_preprocessing': selected_prep,
                    'selected_epoch': selected_epoch,
                    'selected_configuration': config_json
                })
            oof_seed_df = pd.concat([oof_seed_df, pd.DataFrame(rows)], ignore_index=True)
            save_cp(final_metrics_df, CP_FINAL_METRICS)
            save_cp(oof_seed_df, CP_PRED)

            state.update({
                'status': 'IN_PROGRESS',
                'last_completed_outer_fold': outer_fold,
                'last_completed_model': model_name,
                'last_completed_seed': seed,
                'updated_at_utc': datetime.now(timezone.utc).isoformat()
            })
            STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')

            print(f"  seed {seed}: MAE={mm['MAE_days']:.4f} | RMSE={mm['RMSE_days']:.4f} | R2={mm['R2']:.4f} | train={train_s:.1f}s")
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        del pp_outer, Xtr_p, Xte_p, Xtr_m, Xte_m
        gc.collect()

elapsed_this_session = time.perf_counter() - run_start
print(f'\nNB04 main benchmark cell finished in this session: {elapsed_this_session/60:.2f} min')
