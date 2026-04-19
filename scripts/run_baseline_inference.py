#!/usr/bin/env python3
"""
Batch inference for baseline VLMs (Gemma-3, LLaVA-NeXT, InternVL2).

Usage:
    python scripts/run_baseline_inference.py \
        --model gemma3 \
        --test-data data/test_split.jsonl \
        --image-root ~/canonical_dataset/processed_images \
        --output results/gemma3_predictions.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

import torch
import numpy as np

from agri_perceiver.evaluation.baselines import BASELINES, BASELINE_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Baseline VLM inference")
    parser.add_argument("--model", type=str, required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load test data
    samples = []
    with open(args.test_data) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} test samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Resume support
    done_ids = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                item = json.loads(line.strip())
                done_ids.add(item["image_path"])
        print(f"Resuming: {len(done_ids)} already completed")

    # Load baseline model
    print(f"\n=== Loading baseline: {args.model} ===")
    t0 = time.time()
    model_cls = BASELINES[args.model]
    model = model_cls(device="cuda")
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Run inference
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total = len(samples)
    success = 0
    errors = 0
    times = []

    with open(args.output, mode) as fout:
        for i, sample in enumerate(samples):
            img_rel = sample.get("image", "")
            if img_rel in done_ids:
                continue

            img_path = Path(args.image_root) / Path(img_rel).name
            if not img_path.exists():
                img_path = Path(args.image_root) / img_rel
            if not img_path.exists():
                print(f"  [{i+1}/{total}] SKIP missing: {img_rel}")
                errors += 1
                continue

            try:
                t0 = time.time()
                result = model.predict(str(img_path), BASELINE_PROMPT)
                elapsed = time.time() - t0
                times.append(elapsed)
                success += 1

                # Try to extract just the JSON from the output
                output_text = result
                # Strip chat prefixes — find first { and last }
                json_start = output_text.find("{")
                json_end = output_text.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    output_text = output_text[json_start:json_end + 1]

                record = {
                    "image_path": img_rel,
                    "output": output_text,
                    "inference_time_s": round(elapsed, 3),
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                if (i + 1) % 50 == 0 or i == 0:
                    avg_time = np.mean(times[-50:])
                    remaining = total - i - 1 - len(done_ids)
                    eta = avg_time * remaining
                    print(f"  [{i+1}/{total}] {elapsed:.2f}s | avg={avg_time:.2f}s | ETA={eta/60:.0f}min | GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

            except Exception as e:
                errors += 1
                record = {
                    "image_path": img_rel,
                    "output": "",
                    "error": str(e),
                    "inference_time_s": 0.0,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                if errors <= 3:
                    traceback.print_exc()

    print(f"\n=== Done: {success} success, {errors} errors ===")
    if times:
        print(f"Avg inference time: {np.mean(times):.2f}s")
        print(f"Total wall time: {sum(times)/60:.1f} min")


if __name__ == "__main__":
    main()
