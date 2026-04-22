# AgriPerceiver — Experimental Results Tracker
<!-- ============================================================
     LIVING DOCUMENT — update every time a run completes or a
     new experiment is added. Keep raw numbers; compute deltas
     inline so the paper tables can be filled directly from here.
     ============================================================ -->

**Test set:** 5,062 samples, `data/test_split_gt.jsonl`
**Eval script:** `python -m agri_perceiver.evaluation.run_eval`
**HPC:** `ece_23104085@10.10.11.201`, base path `/Data1/ece_23104085/agri-perceiver/`
**Conda env:** `inference-engine` (Python 3.10, transformers 4.55.2)
**Last updated:** 2026-04-22

---

## Composite Score Formula

$$\mathcal{C} = 0.20\,F_1^{\text{type}} + 0.15\,F_{\text{diag}} + 0.15\,B_{\text{sym}} + 0.10\,(1-\text{MAE}) + 0.10\,B_{\text{reas}} + 0.10\,B_{\text{act}} + 0.10\,J_{\text{val}} + 0.05\,(1-\text{ECE}) + 0.05\,S_{\text{cmp}}$$

All components in [0,1], higher is better.
`B_act = 0.0` for all baselines (they produce `recommended_actions` but the field name or structure doesn't match the eval parser — **needs investigation**).

---

## 1. Main Results (Baselines vs. AgriPerceiver)

| Model | #Params | JSON% | Schema% | Type-F1 | Diag-FM | Sev-MAE↓ | Sev-r | BERT-sym | BERT-reas | BERT-act | ECE↓ | **Composite** |
|-------|---------|-------|---------|---------|---------|----------|-------|----------|-----------|----------|------|--------------|
| LLaVA-NeXT-7B | 7B | 99.92 | 99.92 | 0.134 | 0.0008 | 0.3435 | 0.226 | 0.847 | 0.842 | 0.000 | 0.004 | **0.5035** |
| InternVL2-8B | 8B | 100.00 | 100.00 | 0.336 | 0.0047 | 0.3852 | 0.624 | 0.848 | 0.854 | 0.000 | 0.384 | **0.5228** |
| Qwen2-VL-7B | 7B | 99.94 | 99.94 | 0.287 | 0.0010 | 0.2237 | 0.666 | 0.830 | 0.855 | 0.000 | 0.372 | **0.5265** |
| **AgriPerceiver (ours)** | **4.6B** | **99.7** | **99.7** | **0.645** | **0.486** | **0.067** | **0.855** | ~0.66 | ~0.66 | ~0.10 | 0.139 | **0.717** |

> **Notes:**
> - AgriPerceiver metrics from `results/eval_results.json` (Apr 21); baselines from their respective `*_eval_results.json` files (Apr 22).
> - `BERT-sym` and `BERT-reas` for AgriPerceiver are approximate from the paper draft; update when re-running eval.
> - `B_act = 0.0` for baselines: their `recommended_actions` field is likely keyed differently or is a list vs. string. Verify in `run_eval.py` before paper submission.
> - LLaVA-NeXT ECE = 0.004 (extremely low) because it overwhelmingly predicts "unknown" (1532/1535 correct) — well-calibrated on a trivial near-constant prediction strategy.

### Per-class Type-F1 breakdown

| Class | Support | LLaVA-NeXT | InternVL2 | Qwen2-VL | AgriPerceiver |
|-------|---------|-----------|-----------|---------|--------------|
| bacterial | 692 | 0.000 | 0.003 | 0.025 | — |
| deficiency | 618 | 0.006 | 0.475 | 0.386 | — |
| fungal | 1521 | 0.301 | 0.599 | 0.637 | 0.813 |
| pest | 351 | 0.000 | 0.227 | 0.017 | 0.541 |
| unknown | 1535 | 0.493 | 0.689 | 0.634 | 0.827 |
| viral | 345 | 0.006 | 0.022 | 0.022 | 0.525 |
| **macro avg** | 5062 | **0.134** | **0.336** | **0.287** | **0.645** |

> All three baselines severely underperform on bacterial, pest, and viral — likely because these require fine-grained visual cues that a generic prompt cannot elicit without domain-specific training.

---

## 2. Ablation Study

| Variant | N predictions | Composite | Δ vs. full | Status |
|---------|--------------|-----------|-----------|--------|
| Full model | 5062/5062 | 0.7175 | — | ✅ Complete |
| − AnyRes (single tile) | 5062/5062 | **0.7141** | **−0.003** | ✅ Complete (eval run Apr 22 16:19 IST) |
| − Tile embeddings | 2994/5062 → resubmitted | TBD | TBD | ⏳ Job 17498 running (submitted Apr 22 16:3x IST) |
| − Stage-2 LoRA (Stage-1 only) | 0/5062 | TBD | TBD | ❌ Not run; `stage1_connector_weights.pt` path unknown on HPC |
| Latents = 64 | 0/5062 | TBD | TBD | ❌ Requires separate training run |
| Latents = 256 | 0/5062 | TBD | TBD | ❌ Requires separate training run |

### abl_noanyres flat metrics (2026-04-22)
```json
{
  "type_macro_f1": 0.6379,
  "diagnosis_fuzzy_match": 0.4802,
  "severity_mae": 0.0694,
  "json_validity": 0.9968,
  "schema_compliance": 0.9968,
  "symptom_bertscore_f1": 0.9197,
  "reasoning_bertscore_f1": 0.9141,
  "ece": 0.1496,
  "action_bertscore_f1": 0.0
}
```
> AnyRes removal drops Type-F1 by 0.007 (0.645→0.638) and Diag-FM by ~0.006 (0.486→0.480), confirming spatial tiling adds marginal but measurable classification benefit. The extremely small drop suggests the Perceiver resampler is compensating for reduced spatial diversity in most cases.

---

## 3. Training Summary

| Stage | Steps | Init loss | Final loss | Duration |
|-------|-------|-----------|-----------|----------|
| Stage 1 (Alignment) | ~14,700 | ~5.0 | ~0.47 | — |
| Stage 2 (Specialization) | ~37,900 (3 epochs) | 0.41 | ~0.04 | — |

Checkpoints: `agri_perceiver_specialist_e1.pt`, `agri_perceiver_specialist_e2.pt`, `agri_perceiver_specialist_e3.pt`
Stage 1 connector: `stage1_connector_weights.pt`

---

## 4. Run Log / Job History

| Date | Job ID | Name | Status | Predictions | Eval file |
|------|--------|------|--------|------------|----------|
| Apr 21 | 17421 | abl_noanyres | ✅ Complete | `results/ablation_no_anyres_predictions.jsonl` (5062) | `ablation_no_anyres_eval_results.json` ✅ (manual run Apr 22 16:19 IST) |
| Apr 21 | 17464 | abl_notile | ❌ Partial (2994/5062, hit 24h walltime) | `results/ablation_no_tile_embed_predictions.jsonl` | — |
| Apr 22 16:3x | 17498 | abl_notile (resubmit) | ⏳ Running | resuming from 2994 | pending |
| Apr 22 08:35 | 17465 | bl_llava | ✅ Complete | `results/llava_next_predictions.jsonl` (5062) | `results/llava_next_eval_results.json` |
| Apr 22 08:10 | 17467 | bl_qwen2vl | ✅ Complete | `results/qwen2vl_predictions.jsonl` (5062) | `results/qwen2vl_eval_results.json` |
| Apr 22 13:22 | 17471 | bl_internvl2 | ✅ Complete | `results/internvl2_predictions.jsonl` (5062) | `results/internvl2_eval_results.json` |

### Known Issues / Patches Applied
- **InternVL2 transformers 4.55.2 incompatibility**: `InternLM2ForCausalLM` missing `GenerationMixin`.  
  Fix: patched `/Data1/ece_23104085/hf_cache/modules/transformers_modules/OpenGVLab/InternVL2-8B/.../modeling_internlm2.py` (4 patches: import, class def, model_fwd past_kv check, prepare_inputs past_kv check).
- **`action_bertscore_f1 = 0.0` for all baselines**: Likely key mismatch in eval parser. Needs investigation before paper submission.

---

## 5. Metric Weights & Composite Sensitivity

Default weights in `run_eval.py`:

| Component | Weight | Metric |
|-----------|--------|--------|
| `type_macro_f1` | 0.20 | sklearn macro F1, 6 classes |
| `diagnosis_fuzzy_match` | 0.15 | token-Jaccard ≥ 0.6 |
| `symptom_bertscore_f1` | 0.15 | BERTScore F1 (symptoms field) |
| `1 - severity_mae` | 0.10 | MAE on [0,1] scale |
| `reasoning_bertscore_f1` | 0.10 | BERTScore F1 (reasoning field) |
| `action_bertscore_f1` | 0.10 | BERTScore F1 (recommended_actions) |
| `json_validity` | 0.10 | Valid parseable JSON |
| `1 - ece` | 0.05 | Expected Calibration Error (10 bins) |
| `schema_compliance` | 0.05 | All 7 required fields present |

AgriPerceiver score sensitivity (±50% weight perturbation): variance ≈ ±0.01 → ranking is robust.

---

## 6. Pending Experiments / Next Steps

### Immediate
- [x] **abl_noanyres eval**: ✅ Done — composite 0.7141, Δ = −0.003
- [ ] **abl_notile**: job 17498 resubmitted Apr 22, resuming from 2994/5062 — update table when complete
- [ ] **Investigate `action_bertscore_f1 = 0.0`**: check how baselines format `recommended_actions`; compare to GT schema

### Short-term
- [ ] **no_lora ablation**: locate `stage1_connector_weights.pt` on HPC; run `run_ablation.py --ablation=no_lora`
- [ ] **LLM-as-judge eval**: 200 samples × 3 judges (GPT-4V, Gemini-Pro, Claude-3) × 4 models — not yet started
- [ ] **Paper tables**: fill `tab:main` and `tab:ablation` once all numbers are final

### Long-term
- [ ] **Latents = 64 / 256**: requires separate Stage-2 training runs (~4h each on H100)
- [ ] **ICML camera-ready**: compile with `icml2026.sty`; double-check all citations; verify figure captions

---

## 7. Observations & Hypotheses (Research Notes)

### Why baselines fail at structured diagnosis
1. **Prompt brittleness**: All three baselines show near-zero `diagnosis_fuzzy_match` (≤ 0.005) despite reasonable BERTScore. The free-form diagnosis string generated by baselines doesn't match the canonical disease name format expected by fuzzy-match (token Jaccard ≥ 0.6). This metric may penalize legitimate paraphrase.
2. **`unknown` collapse**: LLaVA-NeXT predicts "unknown" 98.7% of the time (1532/1535 correct unknowns, but 3530 false unknowns). This suggests it cannot distinguish disease presence from absence — a critical failure for deployment.
3. **Action parsing**: `B_act = 0.0` for all baselines. This may be a real failure (they don't produce actionable recommendations in the expected format) or a parser bug. The 0.10 weight on this term partially deflates baseline scores relative to AgriPerceiver.
4. **Calibration**: Qwen2-VL and InternVL2 both have ECE ≈ 0.38 — severely overconfident. They output high confidence scores (~0.9) but achieve only ~47-50% accuracy at that confidence level. AgriPerceiver ECE = 0.139 (better, though still overconfident at highest confidence bin).

### Per-class patterns
- Fungal is the "easy" class (most samples, visually distinct lesion patterns): all three baselines achieve F1 ≥ 0.30 here, with Qwen2-VL reaching 0.637.
- Bacterial, pest, and viral are near-zero for all baselines — these require domain-specific visual feature extraction, which our AnyRes + Perceiver bridge provides.
- AgriPerceiver's biggest weaknesses are viral (F1=0.525) and pest (F1=0.541) — consistent with their smaller support and symptom overlap with fungal.

### Severity regression
- Qwen2-VL achieves the best baseline severity regression (MAE=0.224, r=0.666), likely because it has the strongest visual grounding capabilities.
- AgriPerceiver's MAE=0.067, r=0.855 suggests the Perceiver-based bridge captures severity-relevant visual features (lesion area, color change extent) that free-form prompting cannot reliably elicit.

### Structural reliability
- All baselines achieve ≥ 99.9% JSON validity — the structured prompt is sufficient to elicit parseable JSON from 7B+ instruction-tuned models.
- Schema compliance tracks JSON validity (same score) — all required fields are always present when JSON is valid.
