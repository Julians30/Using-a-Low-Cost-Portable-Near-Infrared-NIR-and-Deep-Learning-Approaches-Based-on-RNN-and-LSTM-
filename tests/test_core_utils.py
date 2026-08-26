import numpy as np

from nir_eggs.metrics import regression_metrics
from nir_eggs.preprocessing import SpectralPreprocessor
from nir_eggs.statistics import holm_adjust, kendalls_w_from_friedman, paired_bootstrap_mean_difference


def test_regression_metrics_perfect_prediction():
    y = np.arange(0, 6, dtype=float)
    m = regression_metrics(y, y.copy())
    assert m['MAE_days'] == 0.0
    assert m['RMSE_days'] == 0.0
    assert m['R2'] == 1.0
    assert m['bias_days'] == 0.0
    assert m['within_1d_pct'] == 100.0
    assert m['within_2d_pct'] == 100.0
    assert m['within_3d_pct'] == 100.0


def test_holm_adjust_is_monotone_in_sorted_order():
    raw = np.array([0.01, 0.04, 0.03, 0.20])
    adj = holm_adjust(raw)
    assert np.all((adj >= raw) & (adj <= 1.0))
    order = np.argsort(raw)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_paired_bootstrap_is_deterministic_for_seed():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    r1 = paired_bootstrap_mean_difference(a, b, n_resamples=1000, seed=62026)
    r2 = paired_bootstrap_mean_difference(a, b, n_resamples=1000, seed=62026)
    assert r1 == r2
    assert np.isclose(r1['mean_difference'], 0.5)


def test_kendalls_w_formula_matches_nb06_relation():
    chi2 = 120.19047619047626
    w = kendalls_w_from_friedman(chi2, n_blocks=30, k_treatments=6)
    assert np.isclose(w, 0.8012698412698417)


def test_preprocessor_shapes_and_finiteness():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(12, 331))
    for method in ['raw', 'snv', 'msc', 'sg_smooth', 'sg_deriv1']:
        pp = SpectralPreprocessor(method).fit(X[:8])
        Z = pp.transform(X[8:])
        assert Z.shape == (4, 331)
        assert np.isfinite(Z).all()


def test_msc_reference_is_train_fitted_only():
    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(8, 331))
    X_test = rng.normal(loc=100.0, size=(4, 331))
    pp = SpectralPreprocessor('msc').fit(X_train)
    expected = X_train.mean(axis=0)
    assert np.allclose(pp.msc_reference_, expected)
    _ = pp.transform(X_test)
    assert np.allclose(pp.msc_reference_, expected)
