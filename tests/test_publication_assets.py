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
