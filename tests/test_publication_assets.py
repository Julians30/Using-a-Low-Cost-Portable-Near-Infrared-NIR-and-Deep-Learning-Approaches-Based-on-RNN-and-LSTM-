from pathlib import Path
import json

import numpy as np
import pandas as pd


def test_protocol_and_table3_frozen_values():
    protocol = json.loads(Path("protocol/frozen_protocol.json").read_text())
    assert protocol["dataset"]["independent_eggs"] == 30
    assert protocol["validation"]["outer_folds"] == 5
    assert protocol["statistics"]["primary_inferential_unit"] == "egg"
    assert protocol["external_validation"] is False

    tab = pd.read_csv("tables/publication/Table_3_OOF_Performance.csv")
    tab = tab.set_index("Model")
    assert np.isclose(tab.loc["SVR", "MAE (days)"], 2.194627734359565)
    assert np.isclose(tab.loc["SVR", "RMSE (days)"], 2.716170872669306)
    assert np.isclose(tab.loc["SVR", "R²"], 0.8167059823717481)
    assert np.isclose(tab.loc["PLSR", "MAE (days)"], 2.267298150421753)
    assert np.isclose(tab.loc["ANN", "MAE (days)"], 2.2889400289967803)
    assert tab.loc["BiLSTM", "MAE (days)"] > tab.loc["ANN", "MAE (days)"]
    assert tab.loc["LSTM", "MAE (days)"] > tab.loc["ANN", "MAE (days)"]
    assert tab.loc["SimpleRNN", "MAE (days)"] > tab.loc["ANN", "MAE (days)"]


def test_top_three_posthoc_does_not_support_superiority_claim():
    p = pd.read_csv("tables/publication/Table_4B_Top3_PostHoc.csv")
    assert len(p) == 3
    assert (p["Holm-adjusted p"] >= 0.05).all()


def test_wavelength_order_table_matches_protocol_interpretation():
    t = pd.read_csv("tables/publication/Table_5_Wavelength_Order_Ablation.csv").set_index("Model")
    assert t.loc["SimpleRNN", "Shuffled MAE"] < t.loc["SimpleRNN", "Original MAE"]
    assert t.loc["LSTM", "Shuffled MAE"] < t.loc["LSTM", "Original MAE"]
    assert t.loc["BiLSTM", "Friedman p"] >= 0.05


def test_practical_latency_is_labeled_as_cpu_reference_data():
    t = pd.read_csv("tables/publication/Table_6_Practical_Applicability.csv").set_index("Model")
    assert t.loc["PLSR", "CPU latency (ms/spectrum)"] < t.loc["SVR", "CPU latency (ms/spectrum)"]
    assert t.loc["ANN", "CPU latency (ms/spectrum)"] < t.loc["SimpleRNN", "CPU latency (ms/spectrum)"]
    assert t.loc["SimpleRNN", "CPU latency (ms/spectrum)"] < t.loc["LSTM", "CPU latency (ms/spectrum)"]
    assert t.loc["LSTM", "CPU latency (ms/spectrum)"] < t.loc["BiLSTM", "CPU latency (ms/spectrum)"]


def test_row_level_diagnostic_has_complete_egg_overlap_and_modest_optimism():
    audit = pd.read_csv("results/complementary/row_level_overlap_audit.csv")
    assert len(audit) == 5
    assert (audit["n_overlapping_eggs"] == 30).all()
    assert np.allclose(audit["overlap_fraction_test_eggs"], 1.0)

    comp = pd.read_csv("results/complementary/row_level_vs_eggdisjoint.csv").set_index("model")
    for model in ["SVR", "PLSR", "ANN"]:
        assert comp.loc[model, "MAE_days_rowlevel"] < comp.loc[model, "MAE_days_eggdisjoint"]
        assert comp.loc[model, "R2_rowlevel"] > comp.loc[model, "R2_eggdisjoint"]
    assert 5.0 < comp.loc["ANN", "MAE_relative_reduction_pct"] < 8.0
    assert 5.0 < comp.loc["SVR", "MAE_relative_reduction_pct"] < 8.0
    assert 5.0 < comp.loc["PLSR", "MAE_relative_reduction_pct"] < 8.0


def test_svr_sensitivity_does_not_replace_or_improve_primary_result():
    s = pd.read_csv("results/complementary/svr_sensitivity_summary.csv").set_index("analysis")
    primary = s.loc["Primary frozen NB03 SVR"]
    sens = s.loc["Complementary wider-grid sensitivity SVR"]
    assert sens["MAE_days"] > primary["MAE_days"]
    assert sens["R2"] < primary["R2"]


def test_storage_phase_and_attenuation_outputs_match_manuscript_guardrails():
    phase = pd.read_csv("results/complementary/storage_phase_classification_summary.csv").set_index("model")
    assert np.isclose(phase.loc["SVR", "accuracy"], 0.786364, atol=1e-6)
    assert np.isclose(phase.loc["SVR", "macro_F1"], 0.788418, atol=1e-6)
    assert np.isclose(phase.loc["SVR", "cohen_kappa"], 0.680107, atol=1e-6)

    per_class = pd.read_csv("results/complementary/storage_phase_per_class_metrics.csv")
    assert not per_class["F1"].isna().any()
    srnn_late = per_class[(per_class.model == "SimpleRNN") & (per_class.storage_phase == "Late (15–21)")].iloc[0]
    assert srnn_late["F1"] == 0.0

    slopes = pd.read_csv("results/complementary/predicted_on_observed_attenuation_slopes.csv").set_index("model")
    for model in ["SVR", "PLSR", "ANN"]:
        assert slopes.loc[model, "predicted_on_observed_slope"] < 1.0
        assert slopes.loc[model, "slope_bootstrap95_high"] < 1.0
