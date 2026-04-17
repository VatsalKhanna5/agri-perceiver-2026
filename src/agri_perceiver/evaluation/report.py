"""
Report generation — aggregates results into tables and figures.
"""

import json
from pathlib import Path
from typing import Optional


def generate_markdown_report(
    results: dict,
    model_name: str = "AgriPerceiver",
    baseline_results: Optional[dict[str, dict]] = None,
    output_path: str = "eval_report.md",
) -> str:
    """
    Generate a Markdown evaluation report with tables.

    Args:
        results: Output from run_eval.run_evaluation().
        model_name: Name for the primary model column.
        baseline_results: {baseline_name: results_dict} for comparison.
        output_path: Where to save the report.

    Returns:
        Markdown string.
    """
    lines = [
        "# AgriPerceiver Evaluation Report\n",
        f"**Samples evaluated:** {results.get('n_samples', 'N/A')}  ",
        f"**Composite Score:** {results.get('composite_score', 0):.4f}\n",
    ]

    # Structural metrics
    struct = results.get("structural", {})
    lines.append("## Structural Quality\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| JSON Validity | {struct.get('json_validity', 0):.2%} |")
    lines.append(f"| Schema Compliance | {struct.get('schema_compliance', 0):.2%} |")
    lines.append("")

    # Classification
    cls = results.get("classification", {})
    type_m = cls.get("type", {})
    diag_m = cls.get("diagnosis", {})
    lines.append("## Classification\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| Type Macro F1 | {type_m.get('macro_f1', 0):.4f} |")
    lines.append(f"| Type Weighted F1 | {type_m.get('weighted_f1', 0):.4f} |")
    lines.append(f"| Diagnosis Exact Match | {diag_m.get('exact_match', 0):.2%} |")
    lines.append(f"| Diagnosis Fuzzy Match | {diag_m.get('fuzzy_match', 0):.2%} |")
    lines.append("")

    # Regression
    reg = results.get("regression", {}).get("severity", {})
    lines.append("## Severity Regression\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| MAE | {reg.get('mae', 0):.4f} |")
    lines.append(f"| RMSE | {reg.get('rmse', 0):.4f} |")
    lines.append(f"| Pearson r | {reg.get('pearson_r', 0):.4f} |")
    lines.append("")

    # Calibration
    cal = results.get("calibration", {})
    lines.append("## Calibration\n")
    lines.append(f"**ECE:** {cal.get('ece', 0):.4f}\n")

    # Semantic
    sem = results.get("semantic", {})
    lines.append("## Semantic Similarity (BERTScore)\n")
    lines.append("| Field | F1 |")
    lines.append("|-------|-----|")
    lines.append(f"| Symptoms | {sem.get('symptoms', {}).get('f1', 0):.4f} |")
    lines.append(f"| Reasoning | {sem.get('reasoning', {}).get('f1', 0):.4f} |")
    lines.append("")

    # Baseline comparison table
    if baseline_results:
        lines.append("## Baseline Comparison\n")
        all_models = [model_name] + list(baseline_results.keys())
        header = "| Metric | " + " | ".join(all_models) + " |"
        separator = "|--------|" + "|".join(["-------"] * len(all_models)) + "|"
        lines.append(header)
        lines.append(separator)

        # Composite row
        scores = [f"{results.get('composite_score', 0):.4f}"]
        for bname in baseline_results:
            scores.append(f"{baseline_results[bname].get('composite_score', 0):.4f}")
        lines.append(f"| **Composite** | " + " | ".join(scores) + " |")

        # Key metrics
        for metric_name, path_fn in [
            ("Type F1", lambda r: r.get("classification", {}).get("type", {}).get("macro_f1", 0)),
            ("Diagnosis Match", lambda r: r.get("classification", {}).get("diagnosis", {}).get("fuzzy_match", 0)),
            ("Severity MAE", lambda r: r.get("regression", {}).get("severity", {}).get("mae", 0)),
            ("JSON Valid", lambda r: r.get("structural", {}).get("json_validity", 0)),
            ("ECE", lambda r: r.get("calibration", {}).get("ece", 0)),
        ]:
            vals = [f"{path_fn(results):.4f}"]
            for bname in baseline_results:
                vals.append(f"{path_fn(baseline_results[bname]):.4f}")
            lines.append(f"| {metric_name} | " + " | ".join(vals) + " |")
        lines.append("")

    # Model card info
    lines.append("## Model Architecture\n")
    lines.append("| Component | Detail |")
    lines.append("|-----------|--------|")
    lines.append("| Vision Encoder | SigLIP-SO400M-patch14-384 (frozen) |")
    lines.append("| Perception Bridge | Perceiver Resampler (2 blocks, 128 latents, 28.5× compression) |")
    lines.append("| Language Model | Phi-3-mini-128k-instruct (3.8B, LoRA r=32) |")
    lines.append("| Total Trainable (Stage 1) | ~50M (bridge only) |")
    lines.append("| Total Trainable (Stage 2) | ~85M (bridge + LoRA) |")
    lines.append("| Image Resolution | 5 × 384×384 (AnyRes tiling) |")
    lines.append("")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")

    return report
