#!/usr/bin/env python3
"""
Ablation inference runner for AgriPerceiver VLM.

Supports runtime ablations without retraining:
  - no_lora:   Use Stage 1 connector weights only (no LoRA fine-tuning)
  - no_anyres: Use single center-crop tile instead of 5-tile AnyRes
  - no_tile_embed: Zero out learned spatial tile embeddings

Usage:
    python scripts/run_ablation.py \
        --ablation no_lora \
        --checkpoint checkpoints/stage1_connector_weights.pt \
        --test-data data/test_split.jsonl \
        --image-root ~/canonical_dataset/processed_images \
        --output results/ablation_no_lora_predictions.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agri_perceiver.inference.predictor import (
    AgriPredictor,
    SPECIALIST_PROMPT,
    load_image,
    tile_image,
)


class AblationPredictor:
    """Wraps AgriPredictor with ablation modifications."""

    def __init__(self, predictor: AgriPredictor, ablation: str):
        self.predictor = predictor
        self.ablation = ablation

        if ablation == "no_tile_embed":
            # Zero out the learned spatial tile embeddings
            with torch.no_grad():
                self.predictor.model.tile_embed.tile_embedding.weight.zero_()
            print("[ABLATION] Zeroed tile_embed weights")

    def predict(self, image_path: str) -> str:
        """Run ablation-modified inference, return raw JSON string."""
        img = load_image(image_path)

        if self.ablation == "no_anyres":
            # Single center-crop instead of 5-tile AnyRes
            h, w = img.shape[:2]
            resized = cv2.resize(img, (384, 384))
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            pixel_values = t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 384, 384]
            pixel_values = pixel_values.to(
                self.predictor.device, dtype=torch.bfloat16
            )
        else:
            pixel_values = (
                tile_image(img)
                .unsqueeze(0)
                .to(self.predictor.device, dtype=torch.bfloat16)
            )

        prompt = SPECIALIST_PROMPT
        inputs = self.predictor.tokenizer(prompt, return_tensors="pt").to(
            self.predictor.device
        )
        visual_latents = self.predictor.model.encode_images(pixel_values)

        input_ids = inputs.input_ids
        curr_mask = inputs.attention_mask.to(torch.long)

        generated_tokens = []
        for _ in range(self.predictor.max_new_tokens):
            outputs, _ = self.predictor.model.splice_and_forward(
                input_ids, curr_mask, visual_latents
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            if next_token.item() == self.predictor.tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token.item())
            input_ids = torch.cat(
                [input_ids, next_token.to(self.predictor.device)], dim=1
            )
            curr_mask = torch.cat(
                [
                    curr_mask,
                    torch.ones(
                        (1, 1), device=self.predictor.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )

        return self.predictor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )


def main():
    parser = argparse.ArgumentParser(description="Ablation inference")
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["no_lora", "no_anyres", "no_tile_embed"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
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
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Resume support
    done_ids = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                item = json.loads(line.strip())
                done_ids.add(item["image_path"])
        print(f"Resuming: {len(done_ids)} already completed")

    # Load model
    use_lora = args.ablation != "no_lora"
    print(f"\n=== Loading model (ablation={args.ablation}, lora={use_lora}) ===")

    predictor = AgriPredictor(
        checkpoint_path=args.checkpoint,
        use_lora=use_lora,
    )
    ablation_pred = AblationPredictor(predictor, args.ablation)

    if torch.cuda.is_available():
        print(
            f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated"
        )

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
                errors += 1
                continue

            try:
                t0 = time.time()
                raw_output = ablation_pred.predict(str(img_path))
                elapsed = time.time() - t0
                times.append(elapsed)
                success += 1

                # Extract JSON
                output_text = raw_output
                json_start = output_text.find("{")
                json_end = output_text.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    output_text = output_text[json_start : json_end + 1]

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
                    print(
                        f"  [{i+1}/{total}] {elapsed:.2f}s | "
                        f"avg={avg_time:.2f}s | ETA={eta/60:.0f}min"
                    )

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


if __name__ == "__main__":
    main()
