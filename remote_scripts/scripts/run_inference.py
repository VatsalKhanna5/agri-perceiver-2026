"""
Batch inference on test split — saves predictions JSONL for evaluation.

Usage:
    python scripts/run_inference.py \
        --checkpoint checkpoints/specialist_e3.pt \
        --test-data data/test_split.jsonl \
        --image-root ~/canonical_dataset/processed_images \
        --output results/predictions.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

import torch
import numpy as np

from agri_perceiver.inference.predictor import AgriPredictor, load_image, tile_image, SPECIALIST_PROMPT
from agri_perceiver.inference.schema import DiagnosticReport


def main():
    parser = argparse.ArgumentParser(description="AgriPerceiver batch inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True, help="JSONL with image paths and canonical_report")
    parser.add_argument("--image-root", type=str, required=True, help="Root dir for image paths")
    parser.add_argument("--output", type=str, default="results/predictions.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
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

    # Check for resume
    done_ids = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                item = json.loads(line.strip())
                done_ids.add(item["image_path"])
        print(f"Resuming: {len(done_ids)} already completed")

    # Load model
    print(f"\n=== Loading model from {args.checkpoint} ===")
    t0 = time.time()
    predictor = AgriPredictor(checkpoint_path=args.checkpoint, device="cuda")
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
                # Try the full relative path
                img_path = Path(args.image_root) / img_rel
            if not img_path.exists():
                print(f"  [{i+1}/{total}] SKIP missing: {img_rel}")
                errors += 1
                continue

            try:
                t0 = time.time()
                result = predictor.predict(str(img_path), return_raw=True)
                elapsed = time.time() - t0
                times.append(elapsed)
                success += 1

                record = {
                    "image_path": img_rel,
                    "output": result,
                    "inference_time_s": round(elapsed, 3),
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                if (i + 1) % 50 == 0 or i == 0:
                    avg_time = np.mean(times[-50:])
                    eta = avg_time * (total - i - 1 - len(done_ids))
                    print(f"  [{i+1}/{total}] {elapsed:.2f}s | avg={avg_time:.2f}s | ETA={eta/60:.0f}min | GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

            except Exception as e:
                errors += 1
                record = {
                    "image_path": img_rel,
                    "output": "",
                    "error": str(e),
                    "inference_time_s": 0,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                print(f"  [{i+1}/{total}] ERROR: {e}")

    # Summary
    avg_time = np.mean(times) if times else 0
    total_time = sum(times)
    print(f"\n=== Inference Complete ===")
    print(f"  Samples: {success} success, {errors} errors, {total} total")
    if avg_time > 0:
        print(f"  Avg time: {avg_time:.3f}s/sample ({1/avg_time:.1f} samples/sec)")
    else:
        print("  No timings")
    print(f"  Total inference time: {total_time/60:.1f} min")
    print(f"  Output: {args.output}")

    if torch.cuda.is_available():
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
