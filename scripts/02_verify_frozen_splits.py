#!/usr/bin/env python
"""Stage 02: verify the exact frozen egg-disjoint nested CV assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

from nir_eggs.splits import (
    inner_train_validation_eggs,
    outer_train_test_eggs,
    verify_split_files,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("data/frozen_splits"))
    args = ap.parse_args()

    verify_split_files(args.splits)

    for outer_fold in range(1, 6):
        train, test = outer_train_test_eggs(outer_fold, args.splits)
        assert len(train) == 24 and len(test) == 6
        assert not (set(train) & set(test))
        for inner_fold in range(1, 5):
            itr, iva = inner_train_validation_eggs(
                outer_fold, inner_fold, args.splits
            )
            assert len(itr) == 18 and len(iva) == 6
            assert not (set(itr) & set(iva))
            assert not ((set(itr) | set(iva)) & set(test))
        print(f"outer {outer_fold}: PASS — 24 train / 6 test eggs; 4 inner folds")

    print("PASS — all frozen split hashes and group-disjointness checks succeeded.")


if __name__ == "__main__":
    main()
