"""
Empaqueta las imágenes de melanoma del split de entrenamiento en un ZIP
listo para subir a Google Drive y usar en el notebook de generación de Colab.

Uso:
    python scripts/augmentation/prepare_for_colab.py
"""
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((ROOT / "config/project.yaml").read_text())

SPLITS_DIR = ROOT / cfg["paths"]["data_processed"] / "splits"
OUT_ZIP    = ROOT / cfg["paths"]["data_processed"] / "melanoma_train_for_colab.zip"


def main():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    mel_df   = train_df[train_df["dx"] == "mel"].reset_index(drop=True)

    print(f"Imágenes de melanoma en train: {len(mel_df)}")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in mel_df.iterrows():
            img_path = ROOT / row["image_path"]
            if img_path.exists():
                zf.write(img_path, arcname=f"images/{img_path.name}")

        meta = mel_df[["image_id", "lesion_id", "dx", "label"]].copy()
        zf.writestr("metadata.csv", meta.to_csv(index=False))

        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_split": "train",
            "n_images": len(mel_df),
            "class": "mel",
            "dataset_doi": cfg["dataset"]["doi"],
            "seed": cfg["project"]["seed"],
        }
        zf.writestr("provenance.json", json.dumps(provenance, indent=2))

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"\nZIP creado: {OUT_ZIP.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print("\n--- PRÓXIMOS PASOS ---")
    print("1. Sube el ZIP a Google Drive en la carpeta:")
    print("   Mi unidad / ham10000-augmentation / melanoma_train_for_colab.zip")
    print("2. Abre HAM10000_generation.ipynb en Google Colab")
    print("3. Selecciona Runtime > Change runtime type > T4 GPU")
    print("4. Corre todas las celdas en orden")


if __name__ == "__main__":
    main()
