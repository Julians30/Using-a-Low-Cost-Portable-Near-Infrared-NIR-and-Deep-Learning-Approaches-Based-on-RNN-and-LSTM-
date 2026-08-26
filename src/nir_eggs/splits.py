"""Frozen group-split utilities.

The manuscript's generalization claims use the exact split files in
``data/frozen_splits``. Do not regenerate them during model comparison.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd


EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717"
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root: str | Path = "data/frozen_splits") -> dict:
    root = Path(root)
    path = root / "split_manifest.json"
    observed = sha256_file(path)
    if observed != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen split manifest hash mismatch: {observed} != "
            f"{EXPECTED_SPLIT_MANIFEST_SHA256}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split_files(root: str | Path = "data/frozen_splits") -> None:
    root = Path(root)
    manifest = load_manifest(root)
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Hash mismatch for {name}: {observed} != {expected}")


def outer_assignment(root: str | Path = "data/frozen_splits") -> pd.DataFrame:
    verify_split_files(root)
    return pd.read_csv(Path(root) / "outer_group_assignment_seed2026.csv")


def inner_assignment(
    outer_fold: int, root: str | Path = "data/frozen_splits"
) -> pd.DataFrame:
    if outer_fold not in {1, 2, 3, 4, 5}:
        raise ValueError("outer_fold must be 1..5")
    verify_split_files(root)
    return pd.read_csv(
        Path(root) / f"inner_group_assignment_outer{outer_fold:02d}.csv"
    )


def outer_train_test_eggs(
    outer_fold: int, root: str | Path = "data/frozen_splits"
) -> tuple[list[int], list[int]]:
    assignment = outer_assignment(root)
    test = assignment.loc[assignment["outer_fold"] == outer_fold, "sample"].astype(int)
    train = assignment.loc[assignment["outer_fold"] != outer_fold, "sample"].astype(int)
    if len(train) != 24 or len(test) != 6 or set(train) & set(test):
        raise RuntimeError("Frozen outer split failed integrity checks.")
    return sorted(train.tolist()), sorted(test.tolist())


def inner_train_validation_eggs(
    outer_fold: int,
    inner_fold: int,
    root: str | Path = "data/frozen_splits",
) -> tuple[list[int], list[int]]:
    if inner_fold not in {1, 2, 3, 4}:
        raise ValueError("inner_fold must be 1..4")
    assignment = inner_assignment(outer_fold, root)
    val = assignment.loc[assignment["inner_fold"] == inner_fold, "sample"].astype(int)
    train = assignment.loc[assignment["inner_fold"] != inner_fold, "sample"].astype(int)
    outer_train, outer_test = outer_train_test_eggs(outer_fold, root)
    if set(train) | set(val) != set(outer_train):
        raise RuntimeError("Inner assignment does not cover exactly the outer-training eggs.")
    if set(train) & set(val) or (set(train) | set(val)) & set(outer_test):
        raise RuntimeError("Group leakage detected in frozen inner split.")
    if len(train) != 18 or len(val) != 6:
        raise RuntimeError("Unexpected frozen inner fold size.")
    return sorted(train.tolist()), sorted(val.tolist())
