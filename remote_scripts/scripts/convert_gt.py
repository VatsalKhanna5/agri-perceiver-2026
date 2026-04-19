"""
Convert test_split.jsonl to ground-truth format for run_eval.py.

Input format:  {"image": "...", "canonical_report": {...}, ...}
Output format: {"image_path": "...", "report": {...}}
"""
import json
import sys
from pathlib import Path

input_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_split.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "data/test_split_gt.jsonl"

count = 0
with open(input_path) as fin, open(output_path, "w") as fout:
    for line in fin:
        item = json.loads(line.strip())
        gt_record = {
            "image_path": item["image"],
            "report": item["canonical_report"],
        }
        fout.write(json.dumps(gt_record) + "\n")
        count += 1

print(f"Converted {count} samples: {input_path} -> {output_path}")
