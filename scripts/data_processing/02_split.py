"""
Lesion-aware train/val/test split for HAM10000 binary classification.

Split is performed at the lesion_id level to prevent data leakage.
Outputs three CSV manifests and a metadata JSON to data/processed/splits/.
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((ROOT / "config/project.yaml").read_text())

RAW = ROOT / cfg["dataset"]["download_dir"]
IMAGES_DIR = ROOT / cfg["paths"]["data_processed"] / "images"
SPLITS_DIR = ROOT / cfg["paths"]["data_processed"] / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["project"]["seed"]
TARGET_CLASSES = cfg["split"]["target_classes"]
RATIOS = {
    "train": cfg["split"]["train"],
    "val": cfg["split"]["val"],
    "test": cfg["split"]["test"],
}


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(RAW / "HAM10000_metadata.tab")
    df = meta[meta["dx"].isin(TARGET_CLASSES)].copy()
    df["label"] = (df["dx"] == "mel").astype(int)  # mel=1, nv=0
    df["image_path"] = df["image_id"].apply(
        lambda x: str((IMAGES_DIR / f"{x}.jpg").relative_to(ROOT))
    )
    missing = df[~df["image_path"].apply(lambda p: (ROOT / p).exists())]
    if not missing.empty:
        print(
            f"WARNING: {len(missing)} images referenced in metadata not found on disk.",
            file=sys.stderr,
        )
    return df


def lesion_split(df: pd.DataFrame) -> dict[str, list[str]]:
    lesions = sorted(df["lesion_id"].unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(lesions)

    n = len(lesions)
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])

    return {
        "train": lesions[:n_train],
        "val": lesions[n_train : n_train + n_val],
        "test": lesions[n_train + n_val :],
    }


def verify_no_leakage(splits: dict[str, list[str]]) -> None:
    all_ids = splits["train"] + splits["val"] + splits["test"]
    unique = set()
    for lid in all_ids:
        if lid in unique:
            raise RuntimeError(f"Leakage detected: lesion_id {lid} appears in multiple splits")
        unique.add(lid)
    print("Leakage check: PASSED — no lesion_id crosses splits.")


def save_split(df: pd.DataFrame, lesion_ids: list[str], name: str) -> pd.DataFrame:
    subset = df[df["lesion_id"].isin(lesion_ids)].copy()
    subset = subset[["image_id", "image_path", "lesion_id", "dx", "label"]].reset_index(drop=True)
    out_path = SPLITS_DIR / f"{name}.csv"
    subset.to_csv(out_path, index=False)
    return subset


def main():
    df = load_metadata()
    print(f"Loaded {len(df)} images ({df['lesion_id'].nunique()} unique lesions)")
    print(f"Class distribution:\n{df['dx'].value_counts().to_string()}\n")

    lesion_splits = lesion_split(df)
    verify_no_leakage(lesion_splits)

    stats = {}
    for name, lesion_ids in lesion_splits.items():
        subset = save_split(df, lesion_ids, name)
        stats[name] = {
            "images": len(subset),
            "lesions": len(lesion_ids),
            "mel": int((subset["label"] == 1).sum()),
            "nv": int((subset["label"] == 0).sum()),
        }
        print(
            f"{name:5s}: {stats[name]['images']:5d} images | "
            f"{stats[name]['lesions']:4d} lesions | "
            f"mel={stats[name]['mel']} nv={stats[name]['nv']}"
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_classes": TARGET_CLASSES,
        "ratios": RATIOS,
        "stats": stats,
        "dataset": {
            "name": cfg["dataset"]["name"],
            "doi": cfg["dataset"]["doi"],
            "version": cfg["dataset"]["version"],
            "release_date": cfg["dataset"]["release_date"],
            "source": cfg["dataset"]["source"],
        },
    }
    (SPLITS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSplit manifests saved to {SPLITS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
