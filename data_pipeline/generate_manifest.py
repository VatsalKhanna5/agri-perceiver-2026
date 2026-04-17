"""
Generate image manifest from raw folder structure.

Indexes all images from category-labeled subfolders, assigns unique IDs,
and creates symlinks into a flat processed_images/ directory.

Usage:
    python -m agri_perceiver.data_pipeline.generate_manifest \
        --source canonical_dataset/images \
        --target canonical_dataset/processed_images \
        --output master_manifest.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def generate_manifest(source_dir: Path, target_dir: Path, manifest_path: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue
        label = folder.name
        print(f"Processing: {label}")

        for img_path in sorted(folder.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            unique_id = f"agri_{len(manifest_entries):06d}"
            new_filename = f"{unique_id}{img_path.suffix}"
            target_path = target_dir / new_filename

            entry = {
                "id": unique_id,
                "original_filename": img_path.name,
                "original_label": label,
                "relative_path": str(target_path),
                "status": "pending",
            }

            if not target_path.exists():
                os.symlink(img_path.absolute(), target_path)

            manifest_entries.append(entry)

    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Indexed {len(manifest_entries)} images -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate image manifest")
    parser.add_argument("--source", type=str, required=True, help="Source images directory (with category subfolders)")
    parser.add_argument("--target", type=str, required=True, help="Flat processed images directory")
    parser.add_argument("--output", type=str, required=True, help="Output manifest JSONL path")
    args = parser.parse_args()

    generate_manifest(Path(args.source), Path(args.target), Path(args.output))


if __name__ == "__main__":
    main()
