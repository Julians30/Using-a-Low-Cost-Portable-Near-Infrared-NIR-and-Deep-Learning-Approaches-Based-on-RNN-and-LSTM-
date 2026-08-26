"""Frozen model definitions for the NIR shell-egg storage-time study.

These definitions mirror the final protocol used in NB03/NB04. Architecture
capacity is intentionally fixed for the neural-network models. Hyperparameter
selection for PLSR/SVR must occur only inside the frozen inner egg-disjoint
folds.
"""

from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR


PREPROCESSING_CANDIDATES = ["raw", "snv", "msc", "sg_smooth", "sg_deriv1"]

PLSR_COMPONENT_GRID = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
SVR_C_GRID = [1, 10, 100, 1000, 10000, 100000]
SVR_EPSILON_GRID = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
SVR_GAMMA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

FINAL_SEEDS = [2026, 2027, 2028]

# Frozen NB04 architecture/training capacity.
ANN_HIDDEN = [64, 32]
RECURRENT_UNITS = 64
DENSE_AFTER_RECURRENT = 32
DROPOUT = 0.20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-3
LOSS = "mse"


def build_plsr(n_components: int) -> PLSRegression:
    """Build the PLSR estimator used in the frozen NB03 search/refit."""
    return PLSRegression(n_components=int(n_components), scale=True, max_iter=1000)


def build_svr(C: float, epsilon: float, gamma: float) -> SVR:
    """Build the RBF-SVR estimator used in the frozen chemometric search."""
    return SVR(kernel="rbf", C=float(C), epsilon=float(epsilon), gamma=float(gamma))


def build_deep_model(model_name: str, n_features: int = 331):
    """Build one frozen NB04 neural-network architecture.

    TensorFlow is imported lazily so the lightweight statistical utilities and
    CI checks do not require TensorFlow.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if model_name == "ANN":
        inp = keras.Input(shape=(n_features,), name="spectral_vector")
        x = layers.Dense(ANN_HIDDEN[0], activation="relu")(inp)
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(ANN_HIDDEN[1], activation="relu")(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation="linear")(x)
    else:
        inp = keras.Input(shape=(n_features, 1), name="ordered_wavelength_sequence")
        if model_name == "SimpleRNN":
            x = layers.SimpleRNN(RECURRENT_UNITS, return_sequences=False)(inp)
        elif model_name == "LSTM":
            x = layers.LSTM(RECURRENT_UNITS, return_sequences=False)(inp)
        elif model_name == "BiLSTM":
            x = layers.Bidirectional(
                layers.LSTM(RECURRENT_UNITS, return_sequences=False)
            )(inp)
        else:
            raise ValueError(f"Unknown deep model: {model_name!r}")
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(DENSE_AFTER_RECURRENT, activation="relu")(x)
        x = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inp, out, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def shape_for_deep_model(X, model_name: str):
    """Map a 2-D spectral matrix to the frozen ANN/recurrent input shape."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    if model_name == "ANN":
        return X
    if model_name in {"SimpleRNN", "LSTM", "BiLSTM"}:
        return X[..., None]
    raise ValueError(model_name)
