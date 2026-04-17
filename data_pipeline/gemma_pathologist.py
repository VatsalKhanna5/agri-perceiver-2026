"""
Gemma-3 Agricultural Pathologist — Data Labeling Pipeline.

Uses Gemma-3-12B-IT (4-bit quantized) to generate structured diagnostic
labels for agricultural leaf images. Processes images in batches with
forced JSON pre-fill and structural repair for truncated outputs.

Usage:
    python -m agri_perceiver.data_pipeline.gemma_pathologist \
        --manifest master_manifest.jsonl \
        --output final_agri_dataset.jsonl \
        --batch-size 24

Requires: HF_TOKEN environment variable set for gated model access.
"""

import argparse
import gc
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

PATHOLOGY_CLASSES = (
    "[Healthy, Fungal_Pathogen, Bacterial_Pathogen, Viral_Pathogen, "
    "Pest_Infestation, Nutrient_Deficiency, Abiotic_Stress]"
)


def get_diagnostic_prompt(label: str) -> str:
    """Build the structured diagnostic prompt for Gemma-3."""
    return (
        f"[SYSTEM]: Expert Plant Pathologist & Diagnostic System.\n"
        f"[CROP CONTEXT]: {label}\n\n"
        f"[DIAGNOSTIC HINT]: If this is Citrus/Orange, carefully check for "
        f"ASYMMETRIC blotchy chlorosis (Greening/HLB). Unlike Magnesium "
        f"deficiency, which is usually symmetrical, HLB patterns do not "
        f"mirror across the midrib. If asymmetric, categorize as "
        f"'Bacterial_Pathogen'.\n\n"
        f"[TASK]: Perform a multi-modal clinical audit. Prioritize visual "
        f"evidence over the provided label.\n\n"
        f"[DIAGNOSTIC CRITERIA]:\n"
        f"1. VISUAL CHARACTERISTICS: Evaluate venation, margins, and lesion geometry.\n"
        f"2. PATHOLOGY BASELINE: Categorize strictly into {PATHOLOGY_CLASSES}.\n"
        f"3. CONFIDENCE: Rate certainty (0.0 - 1.0).\n\n"
        f"[STRICT OUTPUT SCHEMA]:\n"
        f"{{\n"
        f"  'visual_features': {{'venation': str, 'margins': str, 'surface_integrity': str}},\n"
        f"  'diagnostic_results': {{\n"
        f"      'classification': str,\n"
        f"      'primary_pathogen': str,\n"
        f"      'severity_index': float,\n"
        f"      'certainty': float\n"
        f"  }},\n"
        f"  'scientific_summary': str,\n"
        f"  'immediate_measures': {{'curative_actions': list, 'preventative_controls': list}}\n"
        f"}}\n"
        f"[STRICT RULE]: If the leaf is vibrant green with no necrotic spots "
        f"or asymmetrical mottling, categorize as 'Healthy'. JSON ONLY."
    )


def process_batch(model, processor, batch_entries: list) -> list:
    """Process a batch of images through Gemma-3."""
    images = [[Image.open(e["relative_path"]).convert("RGB")] for e in batch_entries]
    gc.collect()

    batch_messages = []
    for entry in batch_entries:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": get_diagnostic_prompt(entry["original_label"])},
                ],
            }
        ]
        batch_messages.append(messages)

    prompts = [
        processor.apply_chat_template(msg, add_generation_prompt=True)
        for msg in batch_messages
    ]
    # Forced JSON pre-fill
    prompts = [p + '{"visual_features": {' for p in prompts]

    inputs = processor(
        text=prompts, images=images, return_tensors="pt", padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_tokens = outputs.sequences[:, inputs.input_ids.shape[1] :]
    responses = processor.batch_decode(generated_tokens, skip_special_tokens=True)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    math_confidences = torch.exp(transition_scores).mean(dim=1).tolist()

    results = []
    for i, res in enumerate(responses):
        try:
            text = (
                '{"visual_features": {' + res
                if not res.strip().startswith("{")
                else res
            )
            if text.count("{") > text.count("}"):
                text += "}" * (text.count("{") - text.count("}"))

            parsed = json.loads(text)
            entry = batch_entries[i].copy()
            entry["vlm_analysis"] = parsed
            entry["math_confidence"] = round(math_confidences[i], 4)
            entry["status"] = "completed"
            results.append(entry)
        except Exception as e:
            results.append(
                {**batch_entries[i], "status": "failed", "error": str(e), "raw": res}
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Gemma-3 Agricultural Pathologist")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cleanup-interval", type=int, default=5)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable required for gated model access. "
            "Set it with: export HF_TOKEN=hf_your_token_here"
        )

    from transformers import Gemma3ForConditionalGeneration

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Initializing Gemma-3 12B...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model, token=hf_token)

    with open(args.manifest) as f:
        data = [json.loads(line) for line in f]

    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("status") == "completed":
                        completed_ids.add(record["id"])
                except json.JSONDecodeError:
                    continue

    pending = [e for e in data if e["id"] not in completed_ids]
    print(f"Total: {len(data)} | Completed: {len(completed_ids)} | Pending: {len(pending)}")

    if not pending:
        print("All items already completed.")
        return

    with open(args.output, "a") as out_f:
        for idx in tqdm(range(0, len(pending), args.batch_size), desc="Processing"):
            batch = pending[idx : idx + args.batch_size]

            if (idx // args.batch_size) % args.cleanup_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            try:
                results = process_batch(model, processor, batch)
                for res in results:
                    out_f.write(json.dumps(res) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"\nBatch error at index {idx}: {e}")


if __name__ == "__main__":
    main()
