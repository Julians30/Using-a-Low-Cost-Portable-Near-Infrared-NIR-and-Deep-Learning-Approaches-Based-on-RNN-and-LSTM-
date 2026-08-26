import json
from pathlib import Path


def test_snapshot_matches_frozen_manifests():
    snapshot = json.loads(Path("protocol/reproducibility_snapshot.json").read_text())
    split_manifest = json.loads(Path("data/frozen_splits/split_manifest.json").read_text())
    notebook_manifest = json.loads(Path("notebooks/notebook_source_manifest.json").read_text())

    assert snapshot["frozen_hashes"]["dataset_sha256"] == (
        "cd5021c555ae6b57f892549c574599cef75edf87f58b3f7f4d246ade9327d15e"
    )
    assert snapshot["frozen_hashes"]["split_manifest_sha256"] == (
        "fbeb8fa19d522cd91bee875bf5731cda264475da27bc7e93c25ca0d6f0f33717"
    )
    assert split_manifest["dataset_rows"] == 660
    assert split_manifest["dataset_eggs"] == 30
    assert split_manifest["dataset_days"] == 22

    recorded = {
        item["stage"]: item["clean_sha256"] for item in notebook_manifest["notebooks"]
    }
    assert len(recorded) == 8
    for stage in [f"NB{i:02d}" for i in range(1, 9)]:
        key = f"{stage}_clean_source_sha256"
        assert snapshot["frozen_hashes"][key] == recorded[stage]


def test_snapshot_keeps_external_validation_claim_conservative():
    snapshot = json.loads(Path("protocol/reproducibility_snapshot.json").read_text())
    guard = snapshot["primary_scientific_guardrails"]
    assert guard["independent_biological_unit"] == "egg"
    assert guard["n_independent_eggs"] == 30
    assert guard["n_repeated_spectra"] == 660
    assert guard["outer_test_used_for_tuning"] is False
    assert guard["equivalence_claim_from_non_significance"] is False
    assert guard["external_validation_completed"] is False
