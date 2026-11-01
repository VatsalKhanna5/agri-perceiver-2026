# Evaluation Framework

## Overview

Results for every metric below are in [RESULTS.md](RESULTS.md).

AgriPerceiver's evaluation framework measures model quality across **structural validity**, **classification accuracy**, **regression quality**, **semantic similarity**, **calibration**, and **expert-level judgment** — reflecting the multi-dimensional nature of structured diagnostic output.

## Metric Suite

### 1. Structural Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **JSON Validity** | Fraction of outputs parseable as valid JSON | ≥ 0.95 |
| **Schema Compliance** | Fraction containing all 7 required fields | ≥ 0.90 |

### 2. Classification Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Type Macro F1** | F1 across 6 pathology types (fungal, bacterial, viral, pest, deficiency, unknown) | ≥ 0.70 |
| **Type Weighted F1** | Prevalence-weighted F1 | ≥ 0.75 |
| **Diagnosis Exact Match** | Normalized string equality | ≥ 0.50 |
| **Diagnosis Fuzzy Match** | Token-overlap Jaccard ≥ 0.6 threshold | ≥ 0.65 |

### 3. Regression Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Severity MAE** | Mean absolute error on [0, 1] severity scale | ≤ 0.15 |
| **Severity RMSE** | Root mean squared error | ≤ 0.20 |
| **Pearson r** | Correlation between predicted and true severity | ≥ 0.70 |

### 4. Calibration

| Metric | Description | Target |
|--------|-------------|--------|
| **ECE** | Expected Calibration Error (10 bins) | ≤ 0.10 |

### 5. Semantic Similarity (BERTScore)

| Field | Description | Target F1 |
|-------|-------------|-----------|
| **Symptoms** | Per-symptom best-match BERTScore | ≥ 0.70 |
| **Reasoning** | Full reasoning field comparison | ≥ 0.65 |

### 6. Composite Score

Weighted combination of all metrics (higher = better). The full formula and a robustness check against reweighting are in [RESULTS.md](RESULTS.md#4-composite-score-and-robustness).

| Component | Weight |
|-----------|--------|
| Type F1 | 0.20 |
| Diagnosis Match | 0.15 |
| Symptom BERTScore | 0.15 |
| Severity (1 - MAE) | 0.10 |
| Reasoning BERTScore | 0.10 |
| Action BERTScore | 0.10 |
| JSON Validity | 0.10 |
| Schema Compliance | 0.05 |
| Calibration (1 - ECE) | 0.05 |

## Baseline Comparisons

Three general-purpose baselines with larger parameter counts, implemented in [baselines.py](../src/agri_perceiver/evaluation/baselines.py):

| Model | Key | HuggingFace ID | Parameters |
|-------|-----|----------------|------------|
| **LLaVA-NeXT-7B** | `llava_next` | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B |
| **InternVL2-8B** | `internvl2` | `OpenGVLab/InternVL2-8B` | 8B |
| **Qwen2-VL-7B** | `qwen2vl` | `Qwen/Qwen2-VL-7B-Instruct` | 7B |

Qwen2-VL-7B replaced the originally planned Gemma-3-12B-IT, which is gated; Qwen2-VL is non-gated and of comparable scale. All baselines receive the same prompt asking for structured JSON output and are not fine-tuned. The comparison tests whether a lightweight domain specialist (AgriPerceiver, 4.6B total, 9.2% trained) can beat larger general-purpose models on agricultural pathology.

## LLM-as-Judge Evaluation

Multi-judge consensus system scoring on 5 axes (1-5 each):

1. **Diagnostic Accuracy** — Is the disease identification correct?
2. **Completeness** — Are all symptoms, reasoning, and actions present?
3. **Reasoning Quality** — Is the pathological reasoning sound?
4. **Actionability** — Are treatments practical and correct?
5. **Clinical Reliability** — Would a plant pathologist trust this report?

Inter-judge agreement is measured via score variance across judges.

## Running Evaluation

Install the `eval` extra first (`pip install -e ".[eval]"`). Without `bert-score`, the
BERTScore terms silently come out as 0.

### 1. Batch inference

```bash
python scripts/experiments/run_inference.py     --checkpoint checkpoints/specialist_e3.pt     --test-data data/test_split.jsonl     --image-root /path/to/processed_images     --output results/predictions.jsonl            # add --resume to continue, --max-samples N to smoke-test

python scripts/experiments/validate_predictions.py results/predictions.jsonl   # quick sanity report
```

### 2. Score

```bash
python scripts/experiments/convert_gt.py data/test_split.jsonl data/test_split_gt.jsonl
agri-eval     --predictions results/predictions.jsonl     --ground-truth data/test_split_gt.jsonl     --output results/eval_results.json
```

### 3. Baselines and ablations

```bash
python scripts/experiments/download_baselines.py qwen2vl internvl2 llava_next
python scripts/experiments/run_baseline_inference.py --model qwen2vl     --test-data data/test_split.jsonl --image-root /path/to/processed_images     --output results/qwen2vl_predictions.jsonl

python scripts/experiments/run_ablation.py --ablation no_anyres     --checkpoint checkpoints/specialist_e3.pt     --test-data data/test_split.jsonl --image-root /path/to/processed_images     --output results/ablation_no_anyres_predictions.jsonl
```

InternVL2-8B needs [patch_internlm2.py](../scripts/experiments/patch_internlm2.py) with
transformers ≥ 4.50. Score each predictions file with `agri-eval` as in step 2.

### On the cluster

[jobs/](../jobs/) has a PBS script for each run above: `run_inference.pbs`,
`test10_inference.pbs` (10-sample smoke test), `run_baseline_{llava,internvl,qwen2vl}.pbs` and
`ablation_no_{anyres,lora,tile_embed}.pbs`. The cluster paths in them are site-specific.

### File formats

Predictions JSONL:
```json
{"image_path": "test_images/leaf_001.jpg", "output": "{\"diagnosis\": ...}", "inference_time_s": 1.23}
```

Ground-truth JSONL:
```json
{"image_path": "test_images/leaf_001.jpg", "report": {"diagnosis": "...", "type": "...", ...}}
```
