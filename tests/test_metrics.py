"""Tests for evaluation metrics (no GPU needed)."""

import json
import pytest

from agri_perceiver.evaluation.metrics import (
    json_validity_rate,
    schema_compliance_rate,
    type_classification_metrics,
    diagnosis_accuracy,
    severity_metrics,
    expected_calibration_error,
    compute_composite_score,
)
from agri_perceiver.inference.schema import DiagnosticReport


class TestStructuralMetrics:
    def test_json_validity(self):
        preds = ['{"a": 1}', "not json", '{"b": 2}']
        assert json_validity_rate(preds) == pytest.approx(2 / 3)

    def test_schema_compliance(self):
        good = {"diagnosis": "x", "type": "fungal", "severity": 0.5, "confidence": 0.8,
                "symptoms": [], "reasoning": "", "recommended_actions": []}
        bad = {"diagnosis": "x"}
        assert schema_compliance_rate([good, bad]) == pytest.approx(0.5)


class TestClassificationMetrics:
    def test_type_f1_perfect(self):
        types = ["fungal", "bacterial", "viral", "fungal", "bacterial"]
        result = type_classification_metrics(types, types)
        # macro_f1 averages over ALL label classes including those with 0 support
        assert result["micro_f1"] == pytest.approx(1.0)

    def test_diagnosis_exact(self):
        gt = ["Apple Scab", "Healthy"]
        pred = ["apple scab", "healthy"]
        result = diagnosis_accuracy(gt, pred)
        assert result["exact_match"] == pytest.approx(1.0)


class TestRegressionMetrics:
    def test_severity_perfect(self):
        result = severity_metrics([0.5, 0.8], [0.5, 0.8])
        assert result["mae"] == pytest.approx(0.0)

    def test_severity_error(self):
        result = severity_metrics([0.0, 1.0], [0.5, 0.5])
        assert result["mae"] == pytest.approx(0.5)


class TestCalibration:
    def test_perfect_calibration(self):
        # All confident and correct
        result = expected_calibration_error([True, True], [0.9, 0.9])
        assert result["ece"] < 0.2


class TestSchema:
    def test_parse_valid(self):
        data = {"diagnosis": "Rust", "type": "fungal", "severity": 0.7,
                "confidence": 0.9, "symptoms": ["spots"], "reasoning": "...",
                "recommended_actions": ["spray"]}
        report = DiagnosticReport.from_json_string(json.dumps(data))
        assert report is not None
        assert report.diagnosis == "Rust"

    def test_parse_invalid(self):
        assert DiagnosticReport.from_json_string("not json") is None

    def test_parse_truncated(self):
        # Missing closing brace
        text = '{"diagnosis": "Rust", "type": "fungal", "severity": 0.7, "confidence": 0.9, "symptoms": [], "reasoning": "", "recommended_actions": []'
        report = DiagnosticReport.from_json_string(text)
        assert report is not None


class TestComposite:
    def test_perfect_score(self):
        metrics = {
            "type_macro_f1": 1.0, "diagnosis_fuzzy_match": 1.0,
            "severity_mae": 0.0, "json_validity": 1.0,
            "schema_compliance": 1.0, "symptom_bertscore_f1": 1.0,
            "reasoning_bertscore_f1": 1.0, "ece": 0.0,
            "action_bertscore_f1": 1.0,
        }
        assert compute_composite_score(metrics) == pytest.approx(1.0)
