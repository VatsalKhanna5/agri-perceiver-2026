"""
Evaluation metrics for structured JSON diagnostic reports.

Covers: classification accuracy, regression quality, semantic similarity,
structural validity, calibration, and composite scoring.
All metrics are designed for the agricultural pathology JSON schema.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FieldMetrics:
    """Metrics for a single report field."""
    name: str
    score: float
    count: int
    details: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation result for one prediction."""
    image_id: str
    json_valid: bool
    schema_compliant: bool
    field_metrics: dict = field(default_factory=dict)
    composite_score: float = 0.0


# ---------------------------------------------------------------------------
# 1. Structural Metrics
# ---------------------------------------------------------------------------

def json_validity_rate(predictions: list[str]) -> float:
    """Fraction of predictions that parse as valid JSON."""
    valid = 0
    for pred in predictions:
        try:
            json.loads(pred.strip())
            valid += 1
        except (json.JSONDecodeError, ValueError):
            pass
    return valid / max(len(predictions), 1)


REQUIRED_FIELDS = {"diagnosis", "type", "severity", "confidence", "symptoms", "reasoning", "recommended_actions"}


def schema_compliance_rate(predictions: list[dict]) -> float:
    """Fraction of parsed predictions that contain all required fields."""
    compliant = 0
    for pred in predictions:
        if isinstance(pred, dict) and REQUIRED_FIELDS.issubset(pred.keys()):
            compliant += 1
    return compliant / max(len(predictions), 1)


# ---------------------------------------------------------------------------
# 2. Classification Metrics
# ---------------------------------------------------------------------------

PATHOLOGY_TYPES = {"fungal", "bacterial", "viral", "pest", "deficiency", "unknown"}


def type_classification_metrics(gt_types: list[str], pred_types: list[str]) -> dict:
    """
    Compute macro/micro/weighted F1 for pathology type classification.
    Returns dict with per-class and aggregate metrics.
    """
    from sklearn.metrics import classification_report, f1_score, confusion_matrix

    # Normalize types
    gt_norm = [t.lower().strip() if t else "unknown" for t in gt_types]
    pred_norm = [t.lower().strip() if t else "unknown" for t in pred_types]

    labels = sorted(PATHOLOGY_TYPES)
    report = classification_report(gt_norm, pred_norm, labels=labels, output_dict=True, zero_division=0)

    return {
        "macro_f1": f1_score(gt_norm, pred_norm, labels=labels, average="macro", zero_division=0),
        "micro_f1": f1_score(gt_norm, pred_norm, labels=labels, average="micro", zero_division=0),
        "weighted_f1": f1_score(gt_norm, pred_norm, labels=labels, average="weighted", zero_division=0),
        "per_class": report,
        "confusion_matrix": confusion_matrix(gt_norm, pred_norm, labels=labels).tolist(),
    }


def diagnosis_accuracy(gt_diagnoses: list[str], pred_diagnoses: list[str], threshold: float = 0.6) -> dict:
    """
    Compute diagnosis accuracy using normalized string matching.
    Uses fuzzy matching for partial credit.

    Returns dict with exact_match, fuzzy_match (above threshold), and average similarity.
    """
    exact = 0
    fuzzy = 0
    similarities = []

    for gt, pred in zip(gt_diagnoses, pred_diagnoses):
        gt_clean = _normalize_diagnosis(gt)
        pred_clean = _normalize_diagnosis(pred)

        if gt_clean == pred_clean:
            exact += 1
            fuzzy += 1
            similarities.append(1.0)
        else:
            sim = _token_overlap_similarity(gt_clean, pred_clean)
            similarities.append(sim)
            if sim >= threshold:
                fuzzy += 1

    n = max(len(gt_diagnoses), 1)
    return {
        "exact_match": exact / n,
        "fuzzy_match": fuzzy / n,
        "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
    }


def _normalize_diagnosis(text: str) -> str:
    """Normalize diagnosis string for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[*_`'\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("n/a", "healthy").replace("none", "healthy")
    return text


def _token_overlap_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ---------------------------------------------------------------------------
# 3. Regression Metrics
# ---------------------------------------------------------------------------

def severity_metrics(gt_severities: list[float], pred_severities: list[float]) -> dict:
    """MAE, RMSE, and correlation for severity scores."""
    gt = np.array(gt_severities, dtype=np.float64)
    pred = np.array(pred_severities, dtype=np.float64)

    mae = float(np.mean(np.abs(gt - pred)))
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))

    corr = float(np.corrcoef(gt, pred)[0, 1]) if len(gt) > 1 and np.std(gt) > 0 and np.std(pred) > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "pearson_r": corr}


# ---------------------------------------------------------------------------
# 4. Calibration Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    gt_correct: list[bool], pred_confidences: list[float], n_bins: int = 10
) -> dict:
    """
    Expected Calibration Error (ECE).

    Args:
        gt_correct: Whether the prediction was correct (based on type match).
        pred_confidences: Model's self-reported confidence.
        n_bins: Number of calibration bins.

    Returns:
        ECE score and per-bin calibration data.
    """
    correct = np.array(gt_correct, dtype=np.float64)
    confs = np.array(pred_confidences, dtype=np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi)
        count = mask.sum()

        if count > 0:
            avg_conf = confs[mask].mean()
            avg_acc = correct[mask].mean()
            ece += (count / len(confs)) * abs(avg_acc - avg_conf)
            bin_data.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(count), "avg_conf": float(avg_conf), "avg_acc": float(avg_acc)})

    return {"ece": float(ece), "bins": bin_data}


# ---------------------------------------------------------------------------
# 5. Semantic Similarity Metrics (require bert_score)
# ---------------------------------------------------------------------------

def symptom_recall_bertscore(gt_symptoms_list: list[list[str]], pred_symptoms_list: list[list[str]]) -> dict:
    """
    Compute BERTScore-based symptom recall.
    For each ground truth symptom, find the best-matching predicted symptom.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"error": "bert_score not installed. Install with: pip install bert-score"}

    all_refs = []
    all_cands = []

    for gt_symptoms, pred_symptoms in zip(gt_symptoms_list, pred_symptoms_list):
        if not gt_symptoms or not pred_symptoms:
            continue
        for gt_s in gt_symptoms:
            # Find best match among predicted symptoms
            best_pred = max(pred_symptoms, key=lambda p: len(set(p.lower().split()) & set(gt_s.lower().split())))
            all_refs.append(gt_s)
            all_cands.append(best_pred)

    if not all_refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    P, R, F = bert_score_fn(all_cands, all_refs, lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F.mean()),
    }


def reasoning_similarity(gt_reasoning: list[str], pred_reasoning: list[str]) -> dict:
    """BERTScore for reasoning field comparison."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"error": "bert_score not installed"}

    # Filter empty pairs
    pairs = [(g, p) for g, p in zip(gt_reasoning, pred_reasoning) if g and p]
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    refs, cands = zip(*pairs)
    P, R, F = bert_score_fn(list(cands), list(refs), lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F.mean()),
    }


# ---------------------------------------------------------------------------
# 6. Composite Score
# ---------------------------------------------------------------------------

# Weight allocation for composite score (sums to 1.0)
COMPOSITE_WEIGHTS = {
    "type_f1": 0.20,
    "diagnosis_fuzzy": 0.15,
    "severity_mae_inv": 0.10,  # 1 - MAE
    "json_valid": 0.10,
    "schema_compliant": 0.05,
    "symptom_f1": 0.15,
    "reasoning_f1": 0.10,
    "calibration_inv": 0.05,  # 1 - ECE
    "action_f1": 0.10,
}


def compute_composite_score(metrics: dict) -> float:
    """
    Compute a single composite quality score from individual metrics.
    Each component is normalized to [0, 1] where higher is better.
    """
    score = 0.0

    score += COMPOSITE_WEIGHTS["type_f1"] * metrics.get("type_macro_f1", 0.0)
    score += COMPOSITE_WEIGHTS["diagnosis_fuzzy"] * metrics.get("diagnosis_fuzzy_match", 0.0)
    score += COMPOSITE_WEIGHTS["severity_mae_inv"] * (1.0 - min(metrics.get("severity_mae", 1.0), 1.0))
    score += COMPOSITE_WEIGHTS["json_valid"] * metrics.get("json_validity", 0.0)
    score += COMPOSITE_WEIGHTS["schema_compliant"] * metrics.get("schema_compliance", 0.0)
    score += COMPOSITE_WEIGHTS["symptom_f1"] * metrics.get("symptom_bertscore_f1", 0.0)
    score += COMPOSITE_WEIGHTS["reasoning_f1"] * metrics.get("reasoning_bertscore_f1", 0.0)
    score += COMPOSITE_WEIGHTS["calibration_inv"] * (1.0 - min(metrics.get("ece", 1.0), 1.0))
    score += COMPOSITE_WEIGHTS["action_f1"] * metrics.get("action_bertscore_f1", 0.0)

    return float(score)
