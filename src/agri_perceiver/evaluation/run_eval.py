"""
Master evaluation runner.

Orchestrates: model prediction → metric computation → baseline comparison → report generation.

CLI:
    agri-eval --checkpoint specialist_e3.pt --test-dir test_images/ --output results/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from agri_perceiver.evaluation.metrics import (
    compute_composite_score,
    diagnosis_accuracy,
    expected_calibration_error,
    json_validity_rate,
    reasoning_similarity,
    schema_compliance_rate,
    severity_metrics,
    symptom_recall_bertscore,
    type_classification_metrics,
)
from agri_perceiver.inference.schema import DiagnosticReport


def load_ground_truth(path: str) -> dict[str, dict]:
    """Load ground truth JSONL: each line has {image_path, report}."""
    gt = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            gt[item["image_path"]] = item["report"] if isinstance(item["report"], dict) else json.loads(item["report"])
    return gt


def run_evaluation(
    predictions: dict[str, str],  # image_id -> raw JSON string
    ground_truth: dict[str, dict],  # image_id -> parsed dict
) -> dict:
    """
    Compute all metrics given raw prediction strings and parsed ground truth.

    Returns a comprehensive metrics dict.
    """
    # Align predictions with ground truth
    common_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    if not common_ids:
        return {"error": "No overlapping image IDs between predictions and ground truth."}

    raw_preds = [predictions[k] for k in common_ids]
    gt_dicts = [ground_truth[k] for k in common_ids]

    # Parse predictions
    parsed_preds = []
    for raw in raw_preds:
        report = DiagnosticReport.from_json_string(raw)
        parsed_preds.append(report.model_dump() if report else {})

    # 1. Structural
    json_valid = json_validity_rate(raw_preds)
    schema_comp = schema_compliance_rate(parsed_preds)

    # 2. Classification
    gt_types = [g.get("type", "unknown") for g in gt_dicts]
    pred_types = [p.get("type", "unknown") for p in parsed_preds]
    type_metrics = type_classification_metrics(gt_types, pred_types)

    gt_diag = [g.get("diagnosis", "") for g in gt_dicts]
    pred_diag = [p.get("diagnosis", "") for p in parsed_preds]
    diag_metrics = diagnosis_accuracy(gt_diag, pred_diag)

    # 3. Regression
    gt_sev = [float(g.get("severity", 0)) for g in gt_dicts]
    pred_sev = [float(p.get("severity", 0)) for p in parsed_preds]
    sev_metrics = severity_metrics(gt_sev, pred_sev)

    # 4. Calibration
    gt_correct = [gt_types[i].lower() == pred_types[i].lower() for i in range(len(common_ids))]
    pred_confs = [float(p.get("confidence", 0.5)) for p in parsed_preds]
    cal_metrics = expected_calibration_error(gt_correct, pred_confs)

    # 5. Semantic similarity (may fail if bert_score not installed)
    gt_symptoms = [g.get("symptoms", []) for g in gt_dicts]
    pred_symptoms = [p.get("symptoms", []) for p in parsed_preds]
    sym_metrics = symptom_recall_bertscore(gt_symptoms, pred_symptoms)

    gt_reasoning = [g.get("reasoning", "") for g in gt_dicts]
    pred_reasoning = [p.get("reasoning", "") for p in parsed_preds]
    reas_metrics = reasoning_similarity(gt_reasoning, pred_reasoning)

    # 6. Composite
    flat_metrics = {
        "type_macro_f1": type_metrics["macro_f1"],
        "diagnosis_fuzzy_match": diag_metrics["fuzzy_match"],
        "severity_mae": sev_metrics["mae"],
        "json_validity": json_valid,
        "schema_compliance": schema_comp,
        "symptom_bertscore_f1": sym_metrics.get("f1", 0.0),
        "reasoning_bertscore_f1": reas_metrics.get("f1", 0.0),
        "ece": cal_metrics["ece"],
        "action_bertscore_f1": 0.0,  # TODO: compute separately if needed
    }
    composite = compute_composite_score(flat_metrics)

    return {
        "n_samples": len(common_ids),
        "structural": {"json_validity": json_valid, "schema_compliance": schema_comp},
        "classification": {"type": type_metrics, "diagnosis": diag_metrics},
        "regression": {"severity": sev_metrics},
        "calibration": cal_metrics,
        "semantic": {"symptoms": sym_metrics, "reasoning": reas_metrics},
        "composite_score": composite,
        "flat_metrics": flat_metrics,
    }


def main():
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(description="AgriPerceiver Evaluation Runner")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL (image_path, output)")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSONL")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output path for results")
    args = parser.parse_args()

    # Load data
    gt = load_ground_truth(args.ground_truth)

    preds = {}
    with open(args.predictions) as f:
        for line in f:
            item = json.loads(line.strip())
            preds[item["image_path"]] = item["output"]

    # Run
    results = run_evaluation(preds, gt)
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Composite Score: {results['composite_score']:.4f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
