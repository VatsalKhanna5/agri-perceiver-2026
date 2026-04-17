# Data Pipeline Documentation

## Overview

The AgriPerceiver training data is generated through a three-stage automated pipeline that transforms raw crop disease images into high-quality structured diagnostic labels.

```
Raw Images (57 categories)
    │
    ▼
[1] generate_manifest.py ──→ master_manifest.jsonl (117,635 entries)
    │
    ▼
[2] gemma_pathologist.py ──→ final_agri_dataset.jsonl (raw VLM labels)
    │
    ▼
[3] data_refining.py ──→ final_train_canonical.jsonl (101,301 filtered samples)
```

## Stage 1: Manifest Generation

**Script:** `data_pipeline/generate_manifest.py`

Indexes 117,635 images from 57 category-labeled subfolders under `canonical_dataset/images/`. Each image receives a unique ID (`agri_000000` .. `agri_117634`) and a symlink is created in `canonical_dataset/processed_images/` for flat access.

**Output schema (per line):**
```json
{
  "id": "agri_000042",
  "original_filename": "leaf_042.jpg",
  "original_label": "Apple___Apple_scab",
  "relative_path": "canonical_dataset/processed_images/agri_000042.jpg",
  "status": "pending"
}
```

## Stage 2: VLM Labeling with Gemma-3

**Script:** `data_pipeline/gemma_pathologist.py`

Uses Gemma-3-12B-IT quantized to 4-bit NF4 to analyze each image. Key design choices:

- **Forced JSON pre-fill**: Appends `{"visual_features": {` to the prompt to force structured output
- **Batch processing**: 24 images/batch optimized for H200 VRAM
- **Transition-score confidence**: Computes `math_confidence` from logit transition probabilities
- **Structural repair**: Auto-closes truncated JSON braces
- **Resume-safe**: Appends to output, skips already-completed IDs

**Diagnostic prompt** includes:
- Domain-specific hint (e.g., Citrus HLB asymmetry detection)
- Strict pathology classification taxonomy
- Structured output schema enforcement

**Output schema:**
```json
{
  "id": "agri_000042",
  "vlm_analysis": {
    "visual_features": {"venation": "...", "margins": "...", "surface_integrity": "..."},
    "diagnostic_results": {
      "classification": "Fungal_Pathogen",
      "primary_pathogen": "Venturia inaequalis",
      "severity_index": 0.65,
      "certainty": 0.82
    },
    "scientific_summary": "...",
    "immediate_measures": {"curative_actions": [...], "preventative_controls": [...]}
  },
  "math_confidence": 0.7234,
  "status": "completed"
}
```

## Stage 3: Data Refinement

**Script:** `data_pipeline/data_refining.py`

Filters and transforms raw Gemma-3 output into the canonical training format:

1. **Status filter**: Only `"completed"` entries pass
2. **Manifest join**: Maps IDs to image paths
3. **Confidence threshold**: Drops samples with `math_confidence < 0.5`
4. **Type mapping**: `Fungal_Pathogen → fungal`, `Bacterial_Pathogen → bacterial`, etc.
5. **Bucket assignment**: `easy (≥0.7)`, `medium (0.6-0.7)`, `hard (<0.6)`

**Canonical output schema:**
```json
{
  "image": "canonical_dataset/processed_images/agri_000042.jpg",
  "canonical_report": {
    "diagnosis": "Venturia inaequalis",
    "type": "fungal",
    "severity": 0.65,
    "confidence": 0.82,
    "symptoms": ["prominent venation chlorosis", "irregular margins", "surface lesions"],
    "reasoning": "venation: prominent chlorosis; margins: irregular necrotic edges. ...",
    "recommended_actions": ["Apply copper-based fungicide", "Remove infected leaves"]
  },
  "sample_weight": 0.723,
  "bucket": "easy"
}
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total images indexed | 117,635 |
| Gemma-3 processed | 117,635 |
| After confidence filtering | 101,301 |
| Disease categories | 57 |
| Unique crop species | ~14 |
| Image sources | Kaggle, PlantVillage, open repositories |

## Security Note

The original `gemma_pathologist.py` contained a hardcoded HuggingFace token. The migrated version in `data_pipeline/` reads from the `HF_TOKEN` environment variable instead.
