#!/usr/bin/env python3
"""
Crea data/processed/classification_data.zip con todas las imágenes
(train + val + test, ambas clases) y los CSVs de splits.

Súbelo a Google Drive en:
  Mi unidad/ham10000-augmentation/classification_data.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data/processed/splits"
OUT_ZIP = ROOT / "data/processed/classification_data.zip"


def main() -> None:
    splits = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}

    image_paths: set[Path] = set()
    for df in splits.values():
        for rel in df["image_path"]:
            image_paths.add(ROOT / rel)

    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} imágenes no encontradas. "
            "Asegúrate de haber descargado el dataset con scripts/download_ham10000.py"
        )

    total = len(image_paths)
    print(f"Empaquetando {total} imágenes + 3 CSVs → {OUT_ZIP}")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, path in enumerate(sorted(image_paths), 1):
            zf.write(path, f"images/{path.name}")
            if i % 1000 == 0:
                print(f"  {i}/{total} imágenes")

        for split in ("train", "val", "test"):
            csv_path = SPLITS_DIR / f"{split}.csv"
            zf.write(csv_path, f"splits/{split}.csv")
            print(f"  splits/{split}.csv")

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"\nListo: {OUT_ZIP.name} ({size_mb:.0f} MB)")
    print("Sube este archivo a Drive: Mi unidad/ham10000-augmentation/classification_data.zip")


if __name__ == "__main__":
    main()
