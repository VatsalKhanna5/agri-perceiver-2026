"""
Data refining — Converts raw Gemma-3 output into canonical training format.

Filters by completion status and confidence threshold, then maps
the VLM analysis schema into the canonical {diagnosis, type, severity, ...} format.

Usage:
    python -m agri_perceiver.data_pipeline.data_refining \
        --labels final_agri_dataset.jsonl \
        --manifest master_manifest.jsonl \
        --output final_train_canonical.jsonl
"""

import argparse
import json

from tqdm import tqdm

TYPE_MAPPING = {
    "Pest_Infestation": "pest",
    "Bacterial_Pathogen": "bacterial",
    "Fungal_Pathogen": "fungal",
    "Viral_Pathogen": "viral",
    "Nutrient_Deficiency": "deficiency",
}


def map_type(classification: str) -> str:
    return TYPE_MAPPING.get(classification, "unknown")


def build_reasoning(vlm: dict) -> str:
    summary = vlm.get("scientific_summary", "")
    vf = vlm.get("visual_features", {})
    vf_text = "; ".join(f"{k}: {v}" for k, v in vf.items())
    return f"{vf_text}. {summary}".strip()


def build_actions(vlm: dict) -> list:
    actions = []
    measures = vlm.get("immediate_measures", {})
    for k in measures:
        actions.extend(measures.get(k, []))
    return actions


def assign_bucket(conf: float) -> str:
    if conf >= 0.70:
        return "easy"
    elif conf >= 0.60:
        return "medium"
    return "hard"


def main():
    parser = argparse.ArgumentParser(description="Refine Gemma-3 labels into canonical format")
    parser.add_argument("--labels", type=str, required=True, help="Raw Gemma-3 output JSONL")
    parser.add_argument("--manifest", type=str, required=True, help="Image manifest JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output canonical JSONL")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum math confidence")
    args = parser.parse_args()

    # Load manifest for image paths
    manifest = {}
    with open(args.manifest) as f:
        for line in f:
            entry = json.loads(line)
            manifest[entry["id"].strip()] = entry["relative_path"]

    written = 0
    skipped_failed = 0
    skipped_no_manifest = 0
    skipped_low_conf = 0

    with open(args.labels) as labels, open(args.output, "w") as out:
        for line in tqdm(labels, desc="Refining"):
            sample = json.loads(line)
            sample_id = sample["id"].strip()

            if sample.get("status") != "completed":
                skipped_failed += 1
                continue

            if sample_id not in manifest:
                skipped_no_manifest += 1
                continue

            math_conf = sample.get("math_confidence", 0)
            if math_conf < args.min_confidence:
                skipped_low_conf += 1
                continue

            vlm = sample.get("vlm_analysis", {})
            diag = vlm.get("diagnostic_results", {})

            canonical = {
                "diagnosis": diag.get("primary_pathogen", ""),
                "type": map_type(diag.get("classification", "")),
                "severity": diag.get("severity_index", 0.0),
                "confidence": diag.get("certainty", 0.0),
                "symptoms": list(vlm.get("visual_features", {}).values()),
                "reasoning": build_reasoning(vlm),
                "recommended_actions": build_actions(vlm),
            }

            output_entry = {
                "image": manifest[sample_id],
                "canonical_report": canonical,
                "sample_weight": round(math_conf, 3),
                "bucket": assign_bucket(math_conf),
            }

            out.write(json.dumps(output_entry) + "\n")
            written += 1

    print(f"\nRefinement complete:")
    print(f"  Written: {written}")
    print(f"  Skipped (failed):       {skipped_failed}")
    print(f"  Skipped (no manifest):  {skipped_no_manifest}")
    print(f"  Skipped (low conf):     {skipped_low_conf}")


if __name__ == "__main__":
    main()
