# Publication Figure Captions

**Figure 1.** Leakage-safe primary analytical workflow. The egg is the independent biological unit. All preprocessing, model selection, and epoch selection are restricted to outer-training eggs; outer-test eggs contribute only out-of-fold predictions. Complementary post-freeze sensitivity analyses are described separately in the manuscript.

**Figure 2.** Distribution of the inferential mean per-egg MAE across 30 eggs. The orange line denotes the median, the filled diamond denotes the mean, and open circles denote boxplot outliers. For deep-learning models, the inferential per-egg MAE is calculated within each seed and then averaged across seeds; therefore these values differ slightly from pooled seed-mean OOF MAE used in descriptive performance figures.

**Figure 3.** Observed versus egg-disjoint OOF predicted storage time for SVR, PLSR, and ANN. The dashed line is identity; the solid regression line is predicted storage time regressed on observed storage time. Predicted-on-observed slopes are 0.855 for SVR, 0.845 for PLSR, and 0.823 for ANN and quantify compression toward the center of the experimental range.

**Figure 4.** Wavelength-order ablation for ANN, SimpleRNN, LSTM, and BiLSTM under Original, Reversed, and fixed target-independent Shuffled spectral orders. Categories are intentionally not connected because they are unordered experimental conditions. Error bars show marginal standard errors across eggs; inferential significance is based on paired egg-level tests with Holm correction rather than on visual overlap of marginal error bars.

**Figure 5.** MAE and bias across storage day for the three competitive models using frozen egg-disjoint OOF predictions. Negative bias at later storage days indicates systematic underestimation. Distinct line styles and markers provide redundant encoding in addition to a colorblind-accessible palette.

**Figure 6.** Accuracy–latency reference plane. The vertical metric is pooled OOF MAE computed from seed-mean predictions for deep-learning models; it is descriptive and differs from the inferential per-egg seedwise MAE in Figure 2. The x-axis reports median end-to-end CPU latency per spectrum in the evaluated Colab/TensorFlow implementation on a logarithmic scale. CPU latency is hardware- and implementation-specific and should not be interpreted as SCiO, smartphone, or embedded-device latency.

## Editorial rule

The submission artwork contains **no embedded figure title at the top**. Only axis labels, units, legends, panel labels, and necessary annotations are permitted inside the artwork; captions remain outside the figure in the manuscript. Submission-quality PNG/TIFF files are exported at 600 dpi.
