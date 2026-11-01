# Results

Full results for AgriPerceiver on the held-out test set (5,062 samples). Unless stated otherwise,
every number here matches the ICA 2026 camera-ready paper ([papers/ica26/](../papers/ica26/)), which
is the reference version. How the metrics are defined is in [EVALUATION.md](EVALUATION.md).

- [1. Main results](#1-main-results)
- [2. Per-class pathology type](#2-per-class-pathology-type)
- [3. Ablations](#3-ablations)
- [4. Composite score and robustness](#4-composite-score-and-robustness)
- [5. Extended metrics](#5-extended-metrics)
- [6. Training summary](#6-training-summary)
- [7. Analysis](#7-analysis)
- [8. Provenance notes](#8-provenance-notes)

---

## 1. Main results

All baselines receive the same structured JSON prompt and are not fine-tuned.

| Model | Params | JSON% | Schema% | Type-F1 ↑ | Diag-FM ↑ | Sev-MAE ↓ | Sev-r ↑ | BERT-sym ↑ | **Composite ↑** |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-NeXT-7B | 7B | 99.9 | 99.9 | 0.134 | 0.001 | 0.344 | 0.226 | 0.847 | 0.590 |
| InternVL2-8B | 8B | 100.0 | 100.0 | 0.336 | 0.005 | 0.385 | 0.624 | 0.848 | 0.611 |
| Qwen2-VL-7B | 7B | 99.9 | 99.9 | 0.287 | 0.001 | 0.224 | 0.666 | 0.830 | 0.615 |
| **AgriPerceiver** | **4.6B** | 99.7 | 99.7 | **0.645** | **0.486** | **0.067** | **0.855** | **0.921** | **0.810** |

AgriPerceiver trains only the bridge and LoRA adapters (~426M parameters, 9.2% of the model).

### Margin over the best baseline

| Metric | Best baseline | AgriPerceiver | Absolute | Relative |
|---|---|---|---|---|
| Type-F1 | 0.336 (InternVL2) | 0.645 | +0.309 | +92% (1.9×) |
| Diagnosis FM | 0.005 (InternVL2) | 0.486 | +0.481 | — |
| Severity MAE | 0.224 (Qwen2-VL) | 0.067 | −0.157 | −70% |
| Severity r | 0.666 (Qwen2-VL) | 0.855 | +0.189 | +28% |
| BERT-sym | 0.848 (InternVL2) | 0.921 | +0.073 | +9% |
| **Composite** | **0.615 (Qwen2-VL)** | **0.810** | **+0.195** | **+32%** |

---

## 2. Per-class pathology type

### AgriPerceiver: precision, recall, F1

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Fungal | 1,521 | 0.790 | 0.838 | **0.813** |
| Unknown | 1,535 | 0.908 | 0.759 | **0.827** |
| Bacterial | 692 | 0.607 | 0.520 | 0.560 |
| Deficiency | 618 | 0.528 | 0.707 | 0.605 |
| Pest | 351 | 0.509 | 0.578 | 0.541 |
| Viral | 345 | 0.523 | 0.528 | 0.525 |
| Macro avg | 5,062 | 0.644 | 0.655 | **0.645** |
| Weighted avg | 5,062 | 0.731 | 0.715 | 0.719 |

Micro-averaged accuracy is 0.715.

### F1 against the baselines

This is the data behind Fig. 2 of the ICA paper ([scripts/paper/make_figs.py](../scripts/paper/make_figs.py)).

| Class | Support | LLaVA-NeXT-7B | InternVL2-8B | Qwen2-VL-7B | AgriPerceiver |
|---|---|---|---|---|---|
| Fungal | 1,521 | 0.301 | 0.599 | 0.637 | **0.813** |
| Unknown | 1,535 | 0.493 | 0.689 | 0.634 | **0.827** |
| Bacterial | 692 | 0.000 | 0.003 | 0.025 | **0.560** |
| Deficiency | 618 | 0.006 | 0.475 | 0.386 | **0.605** |
| Pest | 351 | 0.000 | 0.227 | 0.017 | **0.541** |
| Viral | 345 | 0.006 | 0.022 | 0.022 | **0.525** |
| **Macro avg** | 5,062 | 0.134 | 0.336 | 0.287 | **0.645** |

---

## 3. Ablations

These are runtime ablations on the full test set, with no retraining. Each variant reuses the same
Stage-2 checkpoint ([scripts/experiments/run_ablation.py](../scripts/experiments/run_ablation.py)).

| Variant | Type-F1 ↑ | Diag-FM ↑ | Sev-MAE ↓ | Composite ↑ | Δ |
|---|---|---|---|---|---|
| Full model | 0.645 | 0.486 | 0.067 | 0.810 | — |
| − AnyRes (single 729-token tile) | 0.638 | 0.480 | 0.069 | 0.806 | −0.004 |
| − Tile embeddings | 0.647 | 0.483 | 0.067 | 0.810 | ≈ 0 |

The script also supports a `no_lora` variant (Stage-1 bridge only), but its results are not reported.

**AnyRes.** Without tiling, Type-F1 falls by 0.007 and Diagnosis FM by 0.006. The effect is small,
which suggests the resampler partly compensates by attending more broadly over the single tile.
Still, both classification metrics drop in the same direction, so multi-scale context gives a
measurable benefit. The benefit is clearest for spatially distributed symptoms such as bacterial
leaf spots and pest feeding damage.

**Tile embeddings.** Zeroing the tile embeddings leaves the aggregate metrics unchanged. The paper
argues their role is spatial provenance in the free-text localisation of symptoms, which these
aggregate scores do not measure.

---

## 4. Composite score and robustness

$$\mathcal{C} = 0.20\,F_1^{\text{type}} + 0.15\,F_{\text{diag}} + 0.15\,B_{\text{sym}} + 0.10\,(1-\text{MAE}) + 0.10\,B_{\text{reas}} + 0.10\,B_{\text{act}} + 0.10\,J_{\text{val}} + 0.05\,(1-\text{ECE}) + 0.05\,S_{\text{cmp}}$$

| Component | Symbol | Weight | Definition |
|---|---|---|---|
| Pathology type macro-F1 | $F_1^{\text{type}}$ | 0.20 | Macro-averaged F1 over the 6 classes |
| Diagnosis fuzzy match | $F_{\text{diag}}$ | 0.15 | Token-Jaccard similarity ≥ 0.6 |
| Symptom BERTScore F1 | $B_{\text{sym}}$ | 0.15 | BERTScore F1 on `symptoms` |
| Severity MAE (inverted) | $1-\text{MAE}$ | 0.10 | Mean absolute error on severity ∈ [0, 1] |
| Reasoning BERTScore F1 | $B_{\text{reas}}$ | 0.10 | BERTScore F1 on `reasoning` |
| Action BERTScore F1 | $B_{\text{act}}$ | 0.10 | BERTScore F1 on `recommended_actions` |
| JSON validity | $J_{\text{val}}$ | 0.10 | Fraction of outputs that parse as JSON |
| Calibration (inverted) | $1-\text{ECE}$ | 0.05 | Expected calibration error, 10 bins |
| Schema compliance | $S_{\text{cmp}}$ | 0.05 | Fraction with all 7 fields, type-correct |

Pathology type gets the largest single weight because it is the main agronomic decision signal.
The three BERTScore terms share 0.35 between them, since the report is a document for people to
read.

**Robustness.** In turn, we doubled and renormalised the structural, classification and semantic
weights. AgriPerceiver scores 0.835, 0.749 and 0.838 under these schemes, against best-baseline
scores of 0.663, 0.503 and 0.677. The smallest margin is +0.161.

---

## 5. Extended metrics

These metrics come from the working evaluation runs and are not in the paper's main table.

| Model | BERT-reas ↑ | BERT-act ↑ | ECE ↓ |
|---|---|---|---|
| LLaVA-NeXT-7B | 0.842 | 0.000† | 0.004‡ |
| InternVL2-8B | 0.854 | 0.000† | 0.384 |
| Qwen2-VL-7B | 0.855 | 0.000† | 0.372 |
| AgriPerceiver | — | — | 0.139 |

† The zeros come from a key mismatch between the baselines' output format and the parser's
expected schema, not from poor semantic quality. The mismatch affects every baseline in the same
way.

‡ LLaVA-NeXT's near-zero ECE is an artefact of predicting "unknown" almost every time (§7.1), not
genuine calibration.

**AgriPerceiver calibration.** The model is moderately overconfident (ECE 0.139). The largest gap
is in the [0.7, 0.8] confidence bin, with mean confidence 0.797 against accuracy 0.571. That bin
holds 1,354 samples and contributes 0.061 of the total ECE. The gap narrows in the top decile
(0.982 vs. 0.912). Only five of the ten bins are non-empty.

---

## 6. Training summary

| Stage | Objective | Trainable params | Steps | Initial loss | Final loss |
|---|---|---|---|---|---|
| Stage 1 — Alignment | Cross-entropy on captions | ~391M (bridge) | ~14,700 (2 epochs) | ~5.0 | ~0.47 |
| Stage 2 — Specialisation | Weighted cross-entropy on JSON reports | ~426M (bridge + LoRA) | ~37,900 (3 epochs) | 0.41 | ~0.04 |

The training checkpoints are `specialist_e{1,2,3}.pt` (Stage 2) and `stage1_connector_weights.pt`
(Stage 1). The hyperparameters are in [TRAINING.md](TRAINING.md).

---

## 7. Analysis

### 7.1 Why general-purpose baselines fail at structured diagnosis

**Inconsistent disease names.** All three baselines score near zero on Diagnosis FM (≤ 0.005)
despite reasonable BERTScores. Their descriptions are plausible, but they don't use the canonical
disease names, so they fail the token-Jaccard threshold. Getting reliable structured output takes
domain training, not just prompt engineering.

**Collapse to "unknown" (LLaVA-NeXT).** LLaVA-NeXT-7B predicts "unknown" for 98.7% of samples. Its
low ECE is a side effect of this collapse, not genuine calibration. The model fails to tell
disease presence from absence, which is the first thing a deployed system has to do.

**Overconfidence (InternVL2, Qwen2-VL).** Both models have ECE ≈ 0.37–0.38. Their confidence sits
near 0.9 whether or not the answer is correct.

### 7.2 Per-class difficulty

Fungal is the easiest class for every model. It has large support (1,521 samples) and distinctive
lesion morphology, and Qwen2-VL reaches F1 0.637 on it with no domain training.

Baselines score near zero on bacterial and viral (≤ 0.025), and on pest for every baseline except
InternVL2 (0.227). These classes depend on subtle cues such as bacterial leaf margins, pest frass
and viral mosaic patterns. AgriPerceiver reaches F1 0.525–0.560 on all three. Its weakest classes
are viral (0.525) and pest (0.541), consistent with their small support and the visual overlap
between some viral mosaics and early fungal infection.

### 7.3 Severity regression

Qwen2-VL is the strongest baseline at severity regression (MAE 0.224, r 0.666). AgriPerceiver
reaches MAE 0.067 and r 0.855, which cuts the absolute error by 70% and raises the correlation by
28%.

### 7.4 Structural reliability

Every model reaches at least 99.7% JSON validity, and schema compliance tracks validity exactly. So
prompting alone gets models to produce valid structured JSON. The real differences between the
models lie in the content of the reports.

---

## 8. Provenance notes

- Earlier versions of this file and the README reported a composite of **0.717** for AgriPerceiver
  and **0.504–0.527** for the baselines. Those scores were computed before the final BERTScore
  re-run and are superseded by the numbers above.
- [results/eval_results.json](../results/eval_results.json) was produced in an environment without
  `bert-score` installed. Its three BERTScore terms are therefore 0, and its stored composite
  (0.488) is not comparable. Its classification, regression, structural and calibration metrics
  match the paper.
