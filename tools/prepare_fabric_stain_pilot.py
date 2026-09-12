from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

from PIL import Image


DEFAULT_SOURCE = Path(
    r"C:\Users\abdul\Downloads\archive\Fabric Defects Dataset\Fabric Defect Dataset"
)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
AUDIT_REPORT = ROOT / "artifacts" / "fabric_dataset_audit" / "stain_pilot_report.json"
SEED = 230224
NORMAL_DIMS = {(1984, 1488), (1488, 1984)}


def image_info(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        return image.width, image.height, image.mode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_files(source: Path):
    normal_dir = source / "defect free"
    stain_dir = source / "stain"
    if not normal_dir.is_dir() or not stain_dir.is_dir():
        raise FileNotFoundError("Expected 'defect free' and 'stain' folders below the source path.")

    normals = []
    for path in sorted(normal_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        width, height, mode = image_info(path)
        if mode == "L" and (width, height) in NORMAL_DIMS:
            normals.append(path)

    stains = []
    for path in sorted(stain_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        width, height, mode = image_info(path)
        if mode == "L" and (width, height) in NORMAL_DIMS:
            stains.append(path)

    if len(normals) != 68:
        raise ValueError(f"Expected 68 matched grayscale normals; found {len(normals)}.")
    if len(stains) != 398:
        raise ValueError(f"Expected 398 matched grayscale stains; found {len(stains)}.")
    return normals, stains


def build(source: Path, output_zip: Path):
    normals, stains = select_files(source)

    rng = random.Random(SEED)
    shuffled = normals.copy()
    rng.shuffle(shuffled)
    split_files = {
        "train_normal": sorted(shuffled[:48]),
        "val_normal": sorted(shuffled[48:58]),
        "test/good": sorted(shuffled[58:]),
        "test/stain": stains,
    }

    rows = []
    seen_hashes: dict[str, str] = {}
    for split, paths in split_files.items():
        label = "stain" if split == "test/stain" else "good"
        for path in paths:
            width, height, mode = image_info(path)
            digest = sha256(path)
            if digest in seen_hashes:
                raise ValueError(f"Exact duplicate: {path} and {seen_hashes[digest]}")
            seen_hashes[digest] = str(path)
            archive_path = f"fabric_stain_pilot/{split}/{path.name}"
            rows.append(
                {
                    "archive_path": archive_path,
                    "split": split,
                    "label": label,
                    "source_relative_path": str(path.relative_to(source)).replace("\\", "/"),
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )

    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    attribution = """# Fabric stain pilot attribution

This subset was selected from the Fabric Defects Dataset published on Kaggle by
Karthik Vinayan / nexuswho. Dataset citation given by its data card:

Saleem, Saima (2024), "Fabric Defects Dataset", Mendeley Data, V3,
doi:10.17632/663j22s43c.3

Dataset page: https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset
License reported on that page: CC BY 4.0.

The package contains only the matched grayscale defect-free and stain cohorts.
No processed/detected images are included. Defective images are test-only.
The source provides no pixel masks for this stain cohort, so initial evaluation
is image-level plus qualitative anomaly maps. Pixel masks must be annotated for
a later segmentation evaluation.
"""

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for split, paths in split_files.items():
            for path in paths:
                archive.write(path, f"fabric_stain_pilot/{split}/{path.name}")
        archive.writestr("fabric_stain_pilot/manifest.csv", manifest_buffer.getvalue())
        archive.writestr("fabric_stain_pilot/ATTRIBUTION.md", attribution)

    report = {
        "status": "passed",
        "source": str(source),
        "output_zip": str(output_zip),
        "seed": SEED,
        "selection": "grayscale L images at 1984x1488 or 1488x1984",
        "processed_images_included": 0,
        "exact_duplicate_groups": 0,
        "counts": {key: len(value) for key, value in split_files.items()},
        "total": sum(len(value) for value in split_files.values()),
        "limitations": [
            "No pixel masks are supplied for the selected stain cohort.",
            "Filename-level grouping metadata is unavailable; near-duplicate review remains necessary.",
            "This is one stain/capture domain, not a general all-fabric model.",
        ],
    }
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT_ZIP)
    args = parser.parse_args()
    build(args.source, args.output)
