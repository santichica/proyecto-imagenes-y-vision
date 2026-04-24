"""
Extract HAM10000 image ZIPs into data/processed/images/.
Skips files already extracted. Safe to re-run.
"""
import zipfile
from pathlib import Path
import yaml
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((ROOT / "config/project.yaml").read_text())

RAW = ROOT / cfg["dataset"]["download_dir"]
OUT = ROOT / cfg["paths"]["data_processed"] / "images"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_ZIPS = [
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
]


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.filename.endswith("/"):
                continue
            dest = out_dir / Path(member.filename).name
            if dest.exists():
                continue
            zf.extract(member, out_dir)
            # ZipFile extracts with full internal path; flatten to out_dir
            extracted_path = out_dir / member.filename
            if extracted_path != dest:
                extracted_path.rename(dest)
            extracted += 1
    return extracted


def main():
    total = 0
    for name in IMAGE_ZIPS:
        zip_path = RAW / name
        if not zip_path.exists():
            print(f"ERROR: {zip_path} not found", file=sys.stderr)
            sys.exit(1)
        n = extract_zip(zip_path, OUT)
        print(f"{name}: {n} new files extracted → {OUT}")
        total += n

    existing = len(list(OUT.glob("*.jpg")))
    print(f"\nTotal images in {OUT}: {existing}")

    manifest = {
        "source_zips": IMAGE_ZIPS,
        "output_dir": str(OUT.relative_to(ROOT)),
        "image_count": existing,
    }
    (OUT / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
