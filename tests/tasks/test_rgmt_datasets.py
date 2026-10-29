import json

import pytest

from unisupermimic.tasks.motion_tracking.g1.supermimic.datasets import load_motion_split


def test_manifest_resolves_paths_and_rejects_leakage(tmp_path):
    (tmp_path / "first.csv").write_text("first")
    (tmp_path / "second.csv").write_text("second")
    data = {
        "train": [{"path": "first.csv", "source_group": "subject1_take1"}],
        "test": [{"path": "second.csv", "source_group": "subject2_take1"}],
    }
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(data))
    files, identity = load_motion_split(manifest, "test")
    assert files == [str(tmp_path / "second.csv")]
    assert len(identity["motions"][0]["sha256"]) == 64
    data["test"][0]["source_group"] = "subject1_take1"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source_group"):
        load_motion_split(manifest, "test")
    data["test"][0]["source_group"] = "renamed"
    manifest.write_text(json.dumps(data))
    (tmp_path / "second.csv").write_text("first")
    with pytest.raises(ValueError, match="identical motion content"):
        load_motion_split(manifest, "test")
