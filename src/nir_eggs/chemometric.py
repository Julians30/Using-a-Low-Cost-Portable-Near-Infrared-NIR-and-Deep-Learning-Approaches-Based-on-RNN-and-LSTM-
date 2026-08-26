"""Nested PLSR/SVR search helpers for the frozen NIR-HUEVOS protocol."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .metrics import regression_metrics
from .models import build_plsr, build_svr
from .preprocessing import SpectralPreprocessor


def fit_predict_plsr(X_train, y_train, X_test, *, preprocessing, n_components):
    pp = SpectralPreprocessor(preprocessing)
    Xtr = pp.fit_transform(X_train)
    Xte = pp.transform(X_test)
    model = build_plsr(n_components)
    model.fit(Xtr, y_train)
    return model.predict(Xte).ravel()


def fit_predict_svr(X_train, y_train, X_test, *, preprocessing, C, epsilon, gamma):
    # Mirrors the frozen NB03 implementation, including the second train-fitted
    # StandardScaler after the base SpectralPreprocessor.
    pp = SpectralPreprocessor(preprocessing)
    Xtr0 = pp.fit_transform(X_train)
    Xte0 = pp.transform(X_test)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr0)
    Xte = scaler.transform(Xte0)
    model = build_svr(C, epsilon, gamma)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def mean_inner_mae_plsr(folds, *, preprocessing, n_components):
    maes = []
    for Xtr, ytr, Xva, yva in folds:
        pred = fit_predict_plsr(
            Xtr, ytr, Xva,
            preprocessing=preprocessing,
            n_components=n_components,
        )
        maes.append(regression_metrics(yva, pred)["MAE_days"])
    return float(np.mean(maes)), float(np.std(maes, ddof=1))


def mean_inner_mae_svr(folds, *, preprocessing, C, epsilon, gamma):
    maes = []
    for Xtr, ytr, Xva, yva in folds:
        pred = fit_predict_svr(
            Xtr, ytr, Xva,
            preprocessing=preprocessing,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
        )
        maes.append(regression_metrics(yva, pred)["MAE_days"])
    return float(np.mean(maes)), float(np.std(maes, ddof=1))
