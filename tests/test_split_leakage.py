"""
Verifies that the lesion-aware split has no data leakage.
Run after 02_split.py has been executed.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data/processed/splits"


@pytest.fixture(scope="module")
def splits():
    result = {}
    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{name}.csv"
        assert path.exists(), f"{path} not found — run 02_split.py first"
        result[name] = pd.read_csv(path)
    return result


@pytest.fixture(scope="module")
def metadata():
    path = SPLITS_DIR / "metadata.json"
    assert path.exists(), "metadata.json not found — run 02_split.py first"
    return json.loads(path.read_text())


def test_no_lesion_id_leakage(splits):
    """No lesion_id should appear in more than one split."""
    seen = {}
    for split_name, df in splits.items():
        for lid in df["lesion_id"].unique():
            assert lid not in seen, (
                f"Leakage: lesion_id '{lid}' appears in both "
                f"'{seen[lid]}' and '{split_name}'"
            )
            seen[lid] = split_name


def test_no_image_id_leakage(splits):
    """No image_id should appear in more than one split."""
    seen = {}
    for split_name, df in splits.items():
        for iid in df["image_id"].unique():
            assert iid not in seen, (
                f"Leakage: image_id '{iid}' appears in both "
                f"'{seen[iid]}' and '{split_name}'"
            )
            seen[iid] = split_name


def test_only_target_classes(splits, metadata):
    """Splits must contain only the configured target classes."""
    target = set(metadata["target_classes"])
    for split_name, df in splits.items():
        classes = set(df["dx"].unique())
        assert classes <= target, (
            f"Unexpected classes in {split_name}: {classes - target}"
        )


def test_labels_consistent_with_dx(splits):
    """Binary label must be consistent: mel=1, nv=0."""
    for split_name, df in splits.items():
        mel_labels = df[df["dx"] == "mel"]["label"].unique()
        nv_labels = df[df["dx"] == "nv"]["label"].unique()
        assert list(mel_labels) == [1], f"{split_name}: mel label is not 1"
        assert list(nv_labels) == [0], f"{split_name}: nv label is not 0"


def test_split_sizes_reasonable(splits, metadata):
    """Each split should be non-empty and respect rough ratio ordering."""
    sizes = {name: len(df) for name, df in splits.items()}
    assert sizes["train"] > sizes["val"], "train should be larger than val"
    assert sizes["train"] > sizes["test"], "train should be larger than test"
    assert sizes["val"] > 0
    assert sizes["test"] > 0


def test_metadata_has_provenance(metadata):
    """Metadata must record dataset provenance fields."""
    required = {"doi", "name", "version", "release_date", "source"}
    missing = required - set(metadata["dataset"].keys())
    assert not missing, f"Missing provenance fields in metadata: {missing}"


def test_metadata_records_seed(metadata):
    assert "seed" in metadata, "metadata must record the random seed"
