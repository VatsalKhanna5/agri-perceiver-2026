"""
Build a tarball of the 5,062 test images, renamed from original filenames to agri_XXXXXX.jpg.
This is MUCH faster than rglob per-image since we index all files first.
"""
import json
import os
import tarfile
import time
from pathlib import Path

MANIFEST = r"d:\VLM\VLM~\home\vats\master_manifest.jsonl"
TEST_SPLIT = r"d:\VLM\VLM~\home\vats\agri-perceiver\data\test_split.jsonl"
IMAGE_ROOT = Path(r"d:\VLM\VLM~\home\vats\canonical_dataset\images")
OUTPUT_TAR = r"d:\VLM\VLM~\home\vats\agri-perceiver\test_images.tar.gz"

# Step 1: Build manifest mapping (agri_id -> original_filename)
print("Building manifest mapping...")
t0 = time.time()
manifest = {}  # agri_XXXXXX.jpg -> original_filename
with open(MANIFEST) as f:
    for line in f:
        item = json.loads(line.strip())
        agri_id = item["relative_path"].split("/")[-1]
        manifest[agri_id] = item["original_filename"]
print(f"  {len(manifest)} manifest entries in {time.time()-t0:.1f}s")

# Step 2: Get test image names
print("Loading test split...")
test_images = []
with open(TEST_SPLIT) as f:
    for line in f:
        item = json.loads(line.strip())
        img_name = item["image"].split("/")[-1]
        test_images.append(img_name)
print(f"  {len(test_images)} test images")

# Step 3: Index all files under images/ (filename -> full_path)
print("Indexing all image files...")
t0 = time.time()
file_index = {}  # filename -> full path
for dirpath, dirnames, filenames in os.walk(IMAGE_ROOT):
    for fn in filenames:
        file_index[fn] = os.path.join(dirpath, fn)
print(f"  {len(file_index)} files indexed in {time.time()-t0:.1f}s")

# Step 4: Build tarball
print(f"Creating tarball at {OUTPUT_TAR}...")
t0 = time.time()
found = 0
missing = 0
total_size = 0

with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
    for agri_name in test_images:
        orig_name = manifest.get(agri_name)
        if not orig_name:
            print(f"  MISSING in manifest: {agri_name}")
            missing += 1
            continue
        
        full_path = file_index.get(orig_name)
        if not full_path or not os.path.exists(full_path):
            print(f"  MISSING file: {agri_name} -> {orig_name}")
            missing += 1
            continue
        
        fsize = os.path.getsize(full_path)
        if fsize == 0:
            print(f"  EMPTY file: {agri_name} -> {orig_name}")
            missing += 1
            continue
        
        # Add to tar with the agri_XXXXXX.jpg name
        tar.add(full_path, arcname=f"processed_images/{agri_name}")
        total_size += fsize
        found += 1
        
        if found % 1000 == 0:
            print(f"  Added {found} images...")

elapsed = time.time() - t0
tar_size = os.path.getsize(OUTPUT_TAR)
print(f"\n=== Summary ===")
print(f"  Found: {found}, Missing: {missing}")
print(f"  Raw size: {total_size/1e6:.1f} MB")
print(f"  Tar size: {tar_size/1e6:.1f} MB")
print(f"  Time: {elapsed:.1f}s")
print(f"  Output: {OUTPUT_TAR}")
