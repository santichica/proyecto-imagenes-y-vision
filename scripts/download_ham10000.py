#!/usr/bin/env python3
"""Download HAM10000 from Harvard Dataverse using project config."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request


def parse_simple_yaml_block(text: str, block_name: str) -> dict[str, Any]:
    lines = text.splitlines()
    block_start = None
    block_indent = None

    for index, line in enumerate(lines):
        if line.strip() == f"{block_name}:" and not line.startswith(" "):
            block_start = index + 1
            block_indent = None
            break

    if block_start is None:
        raise ValueError(f"Could not find block '{block_name}' in config")

    values: dict[str, Any] = {}
    for line in lines[block_start:]:
        if not line.strip():
            continue
        if not line.startswith(" "):
            break
        if block_indent is None:
            block_indent = len(line) - len(line.lstrip(" "))
        if len(line) - len(line.lstrip(" ")) < block_indent:
            break
        match = re.match(r"^\s{2}([A-Za-z0-9_]+):\s*(.*)$", line)
        if not match:
            continue
        key = match.group(1)
        raw_value = match.group(2).strip()
        if raw_value.startswith(('"', "'")) and raw_value.endswith(('"', "'")):
            value: Any = raw_value[1:-1]
        elif raw_value.isdigit():
            value = int(raw_value)
        else:
            value = raw_value
        values[key] = value
    return values


def read_dataset_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    return parse_simple_yaml_block(text, "dataset")


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent)) as temporary:
        temp_path = Path(temporary.name)
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/project.yaml", help="Path to the project config file")
    parser.add_argument("--include", nargs="*", help="Optional list of filenames to download")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    dataset_config = read_dataset_config(config_path)

    source_url = str(dataset_config["source_url"])
    download_dir = repo_root / str(dataset_config["download_dir"])
    include = set(args.include or [])

    dataset_response = fetch_json(source_url)
    dataset = dataset_response["data"]
    latest_version = dataset["latestVersion"]
    version_label = f'{latest_version["versionNumber"]}.{latest_version["versionMinorNumber"]}'
    downloaded_at = datetime.now(timezone.utc).isoformat()

    files_manifest: list[dict[str, Any]] = []
    for file_entry in latest_version["files"]:
        data_file = file_entry["dataFile"]
        filename = data_file["filename"]
        if include and filename not in include:
            continue

        file_id = data_file["id"]
        file_size = data_file.get("filesize")
        checksum = data_file.get("checksum", {}).get("value")
        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}?format=original"
        destination = download_dir / filename

        should_download = args.force or not destination.exists()
        if not should_download and file_size is not None:
            should_download = destination.stat().st_size != file_size

        if should_download:
            download_file(download_url, destination)

        files_manifest.append(
            {
                "filename": filename,
                "file_id": file_id,
                "checksum_md5": checksum,
                "filesize": file_size,
                "download_url": download_url,
                "local_path": str(destination.relative_to(repo_root)),
            }
        )
        print(f"{filename}: {'downloaded' if should_download else 'skipped'}")

    manifest = {
        "dataset": {
            "source": dataset_config.get("source"),
            "name": dataset_config.get("name"),
            "doi": dataset_config.get("doi"),
            "version": dataset_config.get("version"),
            "release_date": dataset_config.get("release_date"),
            "publication_date": dataset_config.get("publication_date"),
            "source_url": source_url,
            "persistent_url": dataset.get("persistentUrl"),
            "version_state": latest_version.get("versionState"),
            "release_time": latest_version.get("releaseTime"),
            "downloaded_at": downloaded_at,
        },
        "files": files_manifest,
    }

    download_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = download_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
