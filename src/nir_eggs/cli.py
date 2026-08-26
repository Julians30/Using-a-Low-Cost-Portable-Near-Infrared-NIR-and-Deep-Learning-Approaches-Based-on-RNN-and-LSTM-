"""Command-line entry points for lightweight reproducibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import (
    inner_train_validation_eggs,
    outer_train_test_eggs,
    verify_split_files,
)

EXPECTED_DATASET_SHA256 = "cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_splits_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the exact frozen egg-disjoint nested-CV split files."
    )
    parser.add_argument("--splits", type=Path, default=Path("data/frozen_splits"))
    args = parser.parse_args(argv)

    verify_split_files(args.splits)
    all_outer_test = []
    for outer_fold in range(1, 6):
        train, test = outer_train_test_eggs(outer_fold, args.splits)
        all_outer_test.extend(test)
        if len(train) != 24 or len(test) != 6 or set(train) & set(test):
            raise RuntimeError(f"Outer fold {outer_fold} failed integrity checks.")
        for inner_fold in range(1, 5):
            itr, iva = inner_train_validation_eggs(outer_fold, inner_fold, args.splits)
            if len(itr) != 18 or len(iva) != 6:
                raise RuntimeError(
                    f"Outer {outer_fold}, inner {inner_fold}: unexpected group counts."
                )
        print(f"Outer fold {outer_fold}: PASS")

    if sorted(all_outer_test) != list(range(1, 31)):
        raise RuntimeError("Each egg must appear in exactly one outer test fold.")
    print("PASS — frozen split hashes and nested group-disjointness verified.")
    return 0


def audit_data_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the frozen SCiO shell-egg NIR dataset structure."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    observed_hash = _sha256_file(args.data)
    if observed_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            f"Dataset SHA-256 mismatch: {observed_hash} != {EXPECTED_DATASET_SHA256}"
        )

    df = pd.read_csv(args.data)
    spectral_cols = sorted(
        [c for c in df.columns if c.startswith("Spectra_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    wavelengths = np.array([int(c.split("_")[-1]) for c in spectral_cols])

    checks = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "eggs": int(df["sample"].nunique()),
        "storage_days": int(df["storage_days"].nunique()),
        "spectral_variables": int(len(spectral_cols)),
        "wavelength_min_nm": int(wavelengths.min()),
        "wavelength_max_nm": int(wavelengths.max()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dataset_sha256": observed_hash,
    }

    expected = {
        "rows": 660,
        "columns": 333,
        "eggs": 30,
        "storage_days": 22,
        "spectral_variables": 331,
        "wavelength_min_nm": 740,
        "wavelength_max_nm": 1070,
        "missing_values": 0,
        "duplicate_rows": 0,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(f"Dataset audit failed for {key}: {checks[key]} != {value}")

    per_egg = df.groupby("sample")["storage_days"].agg(["size", "nunique", "min", "max"])
    if not (
        per_egg["size"].eq(22).all()
        and per_egg["nunique"].eq(22).all()
        and per_egg["min"].eq(0).all()
        and per_egg["max"].eq(21).all()
    ):
        raise RuntimeError("Dataset is not a complete 30-egg × 22-day repeated-measures panel.")

    checks["status"] = "PASS"
    text = json.dumps(checks, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0
