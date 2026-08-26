#!/usr/bin/env python
"""Stage 01: structural audit of the shell-egg NIR dataset.

This script is the command-line counterpart of NB01. It does not fit models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DATASET_SHA256 = "cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/dataset_egg_storage_RAW.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/data_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    observed_hash = sha256_file(args.data)
    if observed_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: {observed_hash}")

    df = pd.read_csv(args.data)
    spectral_cols = [c for c in df.columns if c.startswith("Spectra_")]
    wavelengths = np.array([int(c.split("_", 1)[1]) for c in spectral_cols])

    assert df.shape == (660, 333)
    assert df["sample"].nunique() == 30
    assert df["storage_days"].nunique() == 22
    assert len(spectral_cols) == 331
    assert wavelengths.min() == 740 and wavelengths.max() == 1070
    assert np.all(np.diff(wavelengths) == 1)
    assert df.isna().sum().sum() == 0
    assert np.isfinite(df[spectral_cols].to_numpy(float)).all()
    assert not df.duplicated().any()

    egg_day = df.groupby(["sample", "storage_days"]).size()
    assert len(egg_day) == 660
    assert egg_day.eq(1).all()

    egg_summary = (
        df.groupby("sample")["storage_days"]
        .agg(n_rows="size", day_min="min", day_max="max", n_unique_days="nunique")
        .reset_index()
    )
    assert egg_summary["n_rows"].eq(22).all()
    assert egg_summary["n_unique_days"].eq(22).all()

    summary = {
        "dataset_sha256": observed_hash,
        "rows": 660,
        "columns": 333,
        "eggs": 30,
        "storage_days": 22,
        "spectral_variables": 331,
        "wavelength_min_nm": 740,
        "wavelength_max_nm": 1070,
        "missing_values": 0,
        "independent_biological_unit": "egg (sample)",
        "status": "PASS",
    }
    (args.out / "dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    egg_summary.to_csv(args.out / "egg_structure.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
