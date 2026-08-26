import numpy as np

from src.nir_eggs.ablation import fixed_shuffled_indices, reorder_after_preprocessing
from src.nir_eggs.models import (
    ANN_HIDDEN,
    BATCH_SIZE,
    DROPOUT,
    FINAL_SEEDS,
    MAX_EPOCHS,
    PLSR_COMPONENT_GRID,
    PREPROCESSING_CANDIDATES,
    RECURRENT_UNITS,
    SVR_C_GRID,
    SVR_EPSILON_GRID,
    SVR_GAMMA_GRID,
)
from src.nir_eggs.splits import inner_train_validation_eggs, outer_train_test_eggs


def test_frozen_model_search_spaces():
    assert PREPROCESSING_CANDIDATES == ["raw", "snv", "msc", "sg_smooth", "sg_deriv1"]
    assert PLSR_COMPONENT_GRID == [2,4,6,8,10,12,15,20,25,30,40,50,60,80,100]
    assert SVR_C_GRID[-1] == 100000
    assert SVR_EPSILON_GRID[-1] == 2.0
    assert SVR_GAMMA_GRID[0] == 1e-5
    assert FINAL_SEEDS == [2026, 2027, 2028]
    assert ANN_HIDDEN == [64, 32]
    assert RECURRENT_UNITS == 64
    assert DROPOUT == 0.20
    assert BATCH_SIZE == 32
    assert MAX_EPOCHS == 300


def test_frozen_outer_and_inner_group_counts():
    all_test = []
    for outer in range(1, 6):
        train, test = outer_train_test_eggs(outer)
        assert len(train) == 24 and len(test) == 6
        assert not (set(train) & set(test))
        all_test.extend(test)
        for inner in range(1, 5):
            itr, iva = inner_train_validation_eggs(outer, inner)
            assert len(itr) == 18 and len(iva) == 6
            assert not (set(itr) & set(iva))
            assert not ((set(itr) | set(iva)) & set(test))
    assert sorted(all_test) == list(range(1, 31))


def test_order_ablation_is_fixed_and_target_independent():
    idx1 = fixed_shuffled_indices(331)
    idx2 = fixed_shuffled_indices(331)
    assert np.array_equal(idx1, idx2)
    assert sorted(idx1.tolist()) == list(range(331))

    X = np.arange(2 * 331).reshape(2, 331)
    assert np.array_equal(reorder_after_preprocessing(X, "original"), X)
    assert np.array_equal(reorder_after_preprocessing(X, "reversed"), X[:, ::-1])
    shuffled = reorder_after_preprocessing(X, "shuffled")
    assert np.array_equal(shuffled, X[:, idx1])
