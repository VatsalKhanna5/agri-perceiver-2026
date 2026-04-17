"""
Pydantic schema for structured diagnostic reports.

Defines the canonical output format that the model is trained to produce.
Also used for output validation and evaluation metric computation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DiagnosticReport(BaseModel):
    """Structured agricultural pathology diagnostic report."""

    diagnosis: str = Field(description="Primary pathogen or condition identified.")
    type: str = Field(description="Pathology class: fungal, bacterial, viral, pest, deficiency, unknown.")
    severity: float = Field(ge=0.0, le=1.0, description="Severity index from 0 (none) to 1 (critical).")
    confidence: float = Field(ge=0.0, le=1.0, description="Model certainty in the diagnosis.")
    symptoms: list[str] = Field(default_factory=list, description="Observed visual symptoms.")
    reasoning: str = Field(default="", description="Pathological reasoning for the diagnosis.")
    recommended_actions: list[str] = Field(default_factory=list, description="Suggested treatments.")

    @classmethod
    def from_json_string(cls, text: str) -> Optional["DiagnosticReport"]:
        """
        Parse a model-generated JSON string into a validated report.
        Returns None if parsing or validation fails.
        """
        import json

        text = text.strip()
        # Handle common model output artifacts
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Attempt structural repair for truncated outputs
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            text += "}" * (open_braces - close_braces)

        try:
            data = json.loads(text)
            return cls(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


# Canonical pathology type classes
PATHOLOGY_TYPES = ["fungal", "bacterial", "viral", "pest", "deficiency", "unknown"]
