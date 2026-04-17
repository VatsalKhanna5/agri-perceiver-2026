"""
LLM-as-Judge evaluator with multi-judge consensus.

Uses multiple LLM judges to score predictions against ground truth
on five axes: diagnostic accuracy, completeness, reasoning quality,
actionability, and clinical reliability.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Evaluation rubric for judges
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a senior agricultural pathologist evaluating AI-generated diagnostic reports.
You will receive a ground truth report and a model prediction for the same leaf image.

Score the prediction on these 5 axes (each 1-5):

1. **Diagnostic Accuracy** (1-5): Is the identified disease correct?
2. **Completeness** (1-5): Are all symptoms, reasoning, and actions present?
3. **Reasoning Quality** (1-5): Is the pathological reasoning sound and specific?
4. **Actionability** (1-5): Are the recommended actions practical and correct?
5. **Clinical Reliability** (1-5): Would a plant pathologist trust this report?

Return ONLY valid JSON:
{
  "diagnostic_accuracy": <int>,
  "completeness": <int>,
  "reasoning_quality": <int>,
  "actionability": <int>,
  "clinical_reliability": <int>,
  "justification": "<brief explanation>"
}
"""

JUDGE_USER_TEMPLATE = """\
**Ground Truth:**
```json
{gt_json}
```

**Model Prediction:**
```json
{pred_json}
```

Score the prediction against the ground truth on the 5 axes (1-5 each)."""


@dataclass
class JudgeScore:
    """Score from a single judge for a single prediction."""
    judge_name: str
    diagnostic_accuracy: int = 0
    completeness: int = 0
    reasoning_quality: int = 0
    actionability: int = 0
    clinical_reliability: int = 0
    justification: str = ""

    @property
    def mean_score(self) -> float:
        scores = [self.diagnostic_accuracy, self.completeness, self.reasoning_quality,
                  self.actionability, self.clinical_reliability]
        return float(np.mean(scores))


@dataclass
class ConsensusScore:
    """Aggregated score from multiple judges."""
    image_id: str
    individual_scores: list[JudgeScore] = field(default_factory=list)

    @property
    def mean_per_axis(self) -> dict:
        if not self.individual_scores:
            return {}
        axes = ["diagnostic_accuracy", "completeness", "reasoning_quality",
                "actionability", "clinical_reliability"]
        return {
            ax: float(np.mean([getattr(s, ax) for s in self.individual_scores]))
            for ax in axes
        }

    @property
    def overall_mean(self) -> float:
        per_axis = self.mean_per_axis
        return float(np.mean(list(per_axis.values()))) if per_axis else 0.0

    @property
    def inter_judge_agreement(self) -> float:
        """Krippendorff-style agreement: 1 - (mean pairwise variance) / (total variance)."""
        if len(self.individual_scores) < 2:
            return 1.0
        means = [s.mean_score for s in self.individual_scores]
        return 1.0 - float(np.var(means))


# ---------------------------------------------------------------------------
# Judge implementations
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Wraps an LLM for judge evaluation.

    Supports any model accessible via transformers or OpenAI-compatible API.
    """

    def __init__(self, name: str, model=None, tokenizer=None, api_fn=None):
        """
        Args:
            name: Judge identifier (e.g., "Gemma-3-12B", "GPT-4o").
            model: HuggingFace model (optional, for local judges).
            tokenizer: HuggingFace tokenizer (optional, for local judges).
            api_fn: Callable(system_prompt, user_prompt) -> str (for API judges).
        """
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.api_fn = api_fn

    def score(self, gt_json: str, pred_json: str) -> JudgeScore:
        """Score a single prediction against ground truth."""
        user_prompt = JUDGE_USER_TEMPLATE.format(gt_json=gt_json, pred_json=pred_json)

        if self.api_fn is not None:
            response = self.api_fn(JUDGE_SYSTEM_PROMPT, user_prompt)
        elif self.model is not None and self.tokenizer is not None:
            response = self._generate_local(user_prompt)
        else:
            raise RuntimeError(f"Judge '{self.name}' has no model or API configured.")

        return self._parse_response(response)

    def _generate_local(self, user_prompt: str) -> str:
        """Generate response using local HuggingFace model."""
        import torch

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _parse_response(self, response: str) -> JudgeScore:
        """Parse judge response into JudgeScore."""
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            data = json.loads(text.strip())
            return JudgeScore(
                judge_name=self.name,
                diagnostic_accuracy=int(data.get("diagnostic_accuracy", 0)),
                completeness=int(data.get("completeness", 0)),
                reasoning_quality=int(data.get("reasoning_quality", 0)),
                actionability=int(data.get("actionability", 0)),
                clinical_reliability=int(data.get("clinical_reliability", 0)),
                justification=str(data.get("justification", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return JudgeScore(judge_name=self.name, justification=f"Parse error: {text[:200]}")


# ---------------------------------------------------------------------------
# Multi-Judge Evaluator
# ---------------------------------------------------------------------------

class MultiJudgeEvaluator:
    """
    Runs multiple LLM judges and computes consensus scores.

    Usage:
        evaluator = MultiJudgeEvaluator([judge1, judge2, judge3])
        consensus = evaluator.evaluate("img_001", gt_json, pred_json)
    """

    def __init__(self, judges: list[LLMJudge]):
        self.judges = judges

    def evaluate(self, image_id: str, gt_json: str, pred_json: str) -> ConsensusScore:
        """Run all judges on a single prediction."""
        scores = []
        for judge in self.judges:
            try:
                score = judge.score(gt_json, pred_json)
                scores.append(score)
            except Exception as e:
                scores.append(JudgeScore(judge_name=judge.name, justification=f"Error: {e}"))

        return ConsensusScore(image_id=image_id, individual_scores=scores)

    def evaluate_batch(
        self,
        image_ids: list[str],
        gt_jsons: list[str],
        pred_jsons: list[str],
    ) -> list[ConsensusScore]:
        """Run all judges on a batch of predictions."""
        return [
            self.evaluate(img_id, gt, pred)
            for img_id, gt, pred in zip(image_ids, gt_jsons, pred_jsons)
        ]

    def summary_statistics(self, results: list[ConsensusScore]) -> dict:
        """Aggregate statistics across all evaluated samples."""
        all_means = [r.overall_mean for r in results]
        axis_means = {}
        for ax in ["diagnostic_accuracy", "completeness", "reasoning_quality",
                    "actionability", "clinical_reliability"]:
            vals = [r.mean_per_axis.get(ax, 0) for r in results]
            axis_means[ax] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        agreements = [r.inter_judge_agreement for r in results]

        return {
            "n_samples": len(results),
            "overall_mean": float(np.mean(all_means)),
            "overall_std": float(np.std(all_means)),
            "per_axis": axis_means,
            "mean_inter_judge_agreement": float(np.mean(agreements)),
        }
