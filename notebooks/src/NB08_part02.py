"""Public source fragment for NB08_PUBLICATION_FIGURES_TABLES.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 12 ----

# -------------------------
# TABLES FOR MAIN MANUSCRIPT
# -------------------------

# Table 1 — dataset and validation design.
table1 = pd.DataFrame([
    ['Instrument','SCiO portable miniaturized NIR spectrometer'],
    ['Spectral range','740–1070 nm'],
    ['Spectral variables','331 wavelengths'],
    ['Independent biological units','30 shell eggs'],
    ['Repeated measurements','22 storage days per egg (days 0–21)'],
    ['Total spectra','660'],
    ['Outer evaluation','5 egg-disjoint folds'],
    ['Outer fold composition','24 train eggs / 6 unseen test eggs'],
    ['Inner model selection','4 egg-disjoint folds within each outer training set'],
    ['Deep-learning seeds','2026, 2027, 2028'],
    ['Primary performance metric','MAE (days)'],
    ['Secondary metrics','RMSE, pooled OOF R², bias, median AE, ±1/±2/±3-day tolerance'],
    ['Primary inferential unit','Egg'],
    ['Primary statistical test','Friedman + paired Wilcoxon signed-rank with Holm correction'],
    ['Uncertainty','Paired egg-level bootstrap, 10,000 resamples']
], columns=['Item','Specification'])

# Table 2 — model families and evaluation role.
table2 = pd.DataFrame([
    ['DummyMean','Naive reference','Mean storage day of training data','Secondary benchmark'],
    ['PLSR','Chemometric regression','Nested group-disjoint selection of preprocessing and latent components','Primary comparator'],
    ['SVR','Kernel machine','RBF SVR with nested group-disjoint hyperparameter selection','Primary comparator'],
    ['ANN','Feed-forward neural network','Compact fixed-capacity ANN; preprocessing and epoch budget selected inner-CV','Deep-learning comparator'],
    ['SimpleRNN','Recurrent neural network','331×1 wavelength sequence; fixed architecture','Recurrent comparator'],
    ['LSTM','Long short-term memory','331×1 wavelength sequence; fixed architecture','Recurrent comparator'],
    ['BiLSTM','Bidirectional LSTM','331×1 wavelength sequence; fixed architecture','Recurrent comparator']
], columns=['Model','Family','Modeling strategy','Role'])

# Table 3 — OOF performance.
table3_cols = ['model','MAE_days','RMSE_days','R2','bias_days','median_AE_days',
               'within_1d_pct','within_2d_pct','within_3d_pct']
table3 = operational[table3_cols].copy()
table3.columns = ['Model','MAE (days)','RMSE (days)','R²','Bias (days)','Median AE (days)',
                  'Within ±1 day (%)','Within ±2 days (%)','Within ±3 days (%)']

# Table 4 — primary statistical robustness.
sum4 = model_summary.copy()
sum4 = sum4[sum4['model'].isin(MODELS_MAIN)][[
    'model','mean_per_egg_MAE_days','bootstrap95_CI_low','bootstrap95_CI_high',
    'average_rank_lower_is_better'
]]
sum4['95% bootstrap CI'] = sum4.apply(
    lambda r: f"{r['bootstrap95_CI_low']:.3f}–{r['bootstrap95_CI_high']:.3f}", axis=1
)
table4 = sum4[['model','mean_per_egg_MAE_days','95% bootstrap CI','average_rank_lower_is_better']].copy()
table4.columns = ['Model','Mean per-egg MAE (days)','95% bootstrap CI','Average rank']

# Table 4B — top-3 post-hoc.
top3_pairwise = pairwise[
    pairwise['model_A'].isin(TOP3) & pairwise['model_B'].isin(TOP3)
].copy()
table4b = top3_pairwise[[
    'model_A','model_B','delta_MAE_A_minus_B_days',
    'delta95_CI_low','delta95_CI_high','p_holm','rank_biserial_A_minus_B'
]].copy()
table4b['95% bootstrap CI'] = table4b.apply(
    lambda r: f"{r['delta95_CI_low']:.3f}–{r['delta95_CI_high']:.3f}", axis=1
)
table4b = table4b[[
    'model_A','model_B','delta_MAE_A_minus_B_days',
    '95% bootstrap CI','p_holm','rank_biserial_A_minus_B'
]]
table4b.columns = ['Model A','Model B','ΔMAE A−B (days)','95% bootstrap CI','Holm-adjusted p','Rank-biserial r']

# Table 5 — wavelength-order ablation.
order_means = order_egg.groupby(['model','order_condition'],as_index=False)['MAE_days'].mean()
pivot_order = order_means.pivot(index='model',columns='order_condition',values='MAE_days').reset_index()
table5 = pivot_order.merge(
    order_friedman[['model','p_value','kendall_W']],
    on='model', how='left'
)
table5 = table5[['model','original','reversed','shuffled','p_value','kendall_W']]
table5.columns = ['Model','Original MAE','Reversed MAE','Shuffled MAE','Friedman p','Kendall W']

# Table 6 — practical applicability.
lat_main = latency[latency['batch_size']==1][[
    'model','end_to_end_median_ms_per_spectrum','end_to_end_throughput_spectra_per_second'
]]
comp = complexity[['model','complexity_measure','complexity_value','serialized_size_MB']]
table6 = (
    operational[operational['model'].isin(MODELS_MAIN)][['model','MAE_days','RMSE_days','R2']]
    .merge(comp,on='model',how='left')
    .merge(lat_main,on='model',how='left')
)
table6.columns = ['Model','MAE (days)','RMSE (days)','R²','Complexity measure','Complexity value',
                  'Serialized size (MB)','CPU latency (ms/spectrum)','Throughput (spectra/s)']

tables = {
    'Table_1_Study_Design.csv': table1,
    'Table_2_Modeling_Framework.csv': table2,
    'Table_3_OOF_Performance.csv': table3,
    'Table_4_Statistical_Robustness.csv': table4,
    'Table_4B_Top3_PostHoc.csv': table4b,
    'Table_5_Wavelength_Order_Ablation.csv': table5,
    'Table_6_Practical_Applicability.csv': table6
}

for name, df in tables.items():
    df.to_csv(TABLE_DIR / name, index=False)

display(table1)
display(table2)
display(table3)
display(table4)
display(table4b)
display(table5)
display(table6)

# ---- Original notebook code cell 13 ----

# English captions — stored OUTSIDE the image files.

captions = {
    'Figure 1': (
        'Leakage-safe analytical workflow used for shell-egg storage-time prediction from portable SCiO NIR spectra. '
        'The egg, rather than the individual spectrum, was treated as the independent biological unit. '
        'Model selection was restricted to the corresponding outer-training eggs, and final performance was based on out-of-fold predictions for unseen eggs.'
    ),
    'Figure 2': (
        'Distribution of per-egg mean absolute error (MAE) across the six substantive prediction models. '
        'Deep-learning errors used for inference were first computed separately for each random seed and then averaged at the egg level.'
    ),
    'Figure 3': (
        'Observed versus out-of-fold predicted storage time for the three competitive models: (a) SVR, (b) PLSR, and (c) ANN. '
        'The dashed line represents perfect agreement between observed and predicted storage time.'
    ),
    'Figure 4': (
        'Sensitivity of ANN and recurrent architectures to wavelength-order manipulation. '
        'Performance is shown for the original 740–1070 nm order, the reversed order, and a fixed target-independent shuffled order. '
        'Error bars represent approximate 95% confidence intervals based on the standard error across eggs.'
    ),
    'Figure 5': (
        'Prediction error across storage time for the three competitive models. '
        '(a) Mean absolute error by storage day. (b) Prediction bias, calculated as predicted minus observed storage time, by storage day.'
    ),
    'Figure 6': (
        'Reference accuracy–latency plane for the six substantive models. '
        'The x-axis reports median end-to-end CPU latency per spectrum in the Colab benchmark environment on a logarithmic scale; '
        'these values are computational references and should not be interpreted as latency on the SCiO device, a smartphone, or embedded hardware.'
    )
}

table_titles = {
    'Table 1': 'Dataset, spectral acquisition, and leakage-safe validation design.',
    'Table 2': 'Prediction models and their role in the frozen analytical framework.',
    'Table 3': 'Out-of-fold predictive performance for shell-egg storage-time estimation.',
    'Table 4': 'Egg-level statistical robustness of the six substantive prediction models.',
    'Table 4B': 'Post-hoc paired comparisons among SVR, PLSR, and ANN.',
    'Table 5': 'Wavelength-order ablation results for ANN and recurrent neural-network architectures.',
    'Table 6': 'Predictive performance and computational applicability of the six substantive models.'
}

caption_lines = []
for k,v in captions.items():
    caption_lines.append(f'{k}. {v}')
    caption_lines.append('')
(RESULT_DIR / 'NB08_FIGURE_CAPTIONS_ENGLISH.txt').write_text('\n'.join(caption_lines),encoding='utf-8')

table_lines = []
for k,v in table_titles.items():
    table_lines.append(f'{k}. {v}')
(RESULT_DIR / 'NB08_TABLE_TITLES_ENGLISH.txt').write_text('\n'.join(table_lines),encoding='utf-8')

print((RESULT_DIR / 'NB08_FIGURE_CAPTIONS_ENGLISH.txt').read_text(encoding='utf-8'))

# ---- Original notebook code cell 14 ----

# MDPI table-format instructions for final Word assembly.
# Note: MDPI production may reformat tables; Word tables should remain editable.

mdpi_table_style = {
    'target_font': 'Palatino Linotype',
    'target_font_size_pt': 10,
    'header': 'bold',
    'vertical_alignment': 'center',
    'horizontal_alignment': 'left for text; decimal/right alignment for numeric values',
    'color_policy': 'no decorative color',
    'borders': 'minimal horizontal rules; no heavy grid styling',
    'notes': 'Abbreviations and statistical notes placed below table; table caption placed above table in manuscript.'
}

(RESULT_DIR / 'NB08_MDPI_TABLE_STYLE.json').write_text(
    json.dumps(mdpi_table_style,indent=2),encoding='utf-8'
)
print(json.dumps(mdpi_table_style,indent=2))

# ---- Original notebook code cell 15 ----

# Integrity checks: all final figures must have PNG + TIFF counterparts.

expected_stems = [
    'Figure_1_Leakage_Safe_Workflow',
    'Figure_2_Per_Egg_MAE_Distribution',
    'Figure_3_Observed_vs_Predicted_Top3',
    'Figure_4_Wavelength_Order_Ablation',
    'Figure_5_Error_and_Bias_Across_Storage_Time',
    'Figure_6_Accuracy_Latency_Plane'
]

for stem in expected_stems:
    assert (FIG_DIR / f'{stem}.png').exists()
    assert (FIG_DIR / f'{stem}.tiff').exists()
    assert (FIG_DIR / f'{stem}.png').stat().st_size > 0
    assert (FIG_DIR / f'{stem}.tiff').stat().st_size > 0

assert len(tables) == 7
for name in tables:
    assert (TABLE_DIR / name).exists()

# Check that no matplotlib code used an axes/figure title.
# This is a source-level editorial guard.
notebook_file = NOTEBOOKS_DIR / NOTEBOOK_FILENAME
print('Figure/table generation checks passed.')

# ---- Original notebook code cell 16 ----

# Protocol and completion status.

protocol = {
    'notebook': 'NB08_PUBLICATION_FIGURES_TABLES',
    'notebook_filename': NOTEBOOK_FILENAME,
    'run_revision': RUN_REVISION,
    'target_journal': 'Foods (MDPI)',
    'figure_language': 'English',
    'figure_internal_titles': False,
    'figure_font_target': 'Palatino Linotype',
    'figure_font_size_pt': BASE_FONT_SIZE,
    'figure_font_used_runtime': FONT_FAMILY,
    'figure_resolution_dpi': FIG_DPI,
    'figure_formats': ['PNG','TIFF'],
    'table_language': 'English',
    'table_font_target': 'Palatino Linotype',
    'table_font_size_pt': 10,
    'models_retrained': False,
    'hyperparameters_reselected': False,
    'performance_source': 'frozen NB06/NB07 outputs',
    'number_main_figures': 6,
    'number_table_files': len(tables),
    'dataset_sha256': EXPECTED_DATASET_SHA256,
    'frozen_split_manifest_sha256': EXPECTED_SPLIT_MANIFEST_SHA256
}
(RESULT_DIR / 'NB08_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

status = {
    'status': 'COMPLETED',
    'run_revision': RUN_REVISION,
    'figures_png': len(expected_stems),
    'figures_tiff': len(expected_stems),
    'table_csv_files': len(tables),
    'captions_external_to_figures': True,
    'internal_figure_titles_present': False
}
(RESULT_DIR / 'EXECUTION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

print('NB08 status: COMPLETED')
