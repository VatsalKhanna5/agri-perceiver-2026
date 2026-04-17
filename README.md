# AgriPerceiver VLM

**A Perceiver-based Vision-Language Model for Agricultural Pathology Diagnosis**

> Submitted to ICML 2026 — 3rd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences

AgriPerceiver is a lightweight, domain-specialized VLM that takes a photograph of a plant leaf and generates a structured JSON diagnostic report covering disease identification, pathology type, severity assessment, visual symptom analysis, pathological reasoning, and treatment recommendations.

## Key Contributions

- **28.5× visual token compression** via a Perceiver Resampler bridge (3,645 → 128 latents), enabling efficient multimodal fusion without sacrificing spatial fidelity
- **Backbone-agnostic architecture** — the perception bridge is decoupled from specific vision/language model choices
- **Two-stage curriculum** (alignment → specialization) with only ~2% trainable parameters in Stage 2 (LoRA)
- **Automated data labeling pipeline** using Gemma-3-12B-IT to generate 101K structured labels from raw images
- **Comprehensive evaluation framework** with 9 metrics, 3 baselines, and LLM-as-judge consensus scoring

## Architecture

```
Image (H×W×3)
  │
  ├─ AnyRes Tiling ──→ 5 tiles (4 quadrants + 1 global) × 384×384
  │
  ├─ SigLIP-SO400M ──→ [5, 729, 1152] patch features (frozen)
  │
  ├─ TileEmbeddings ──→ + learned spatial position per tile
  │
  ├─ VisionProjector ──→ MLP 1152 → 3072 (match LLM dim)
  │
  ├─ PerceiverResampler ──→ 3,645 tokens → 128 latents (28.5× compression)
  │
  └─ Splice into Phi-3 ──→ [<image_start> + 128 latents + <image_end>]
                              │
                              ▼
                     Phi-3-mini-128k-instruct (3.8B, LoRA r=32)
                              │
                              ▼
                     Structured JSON diagnostic report
```

| Component | Model | Parameters |
|-----------|-------|------------|
| Vision Encoder | SigLIP-SO400M-patch14-384 | ~400M (frozen) |
| Perception Bridge | Perceiver + Projector + TileEmbed | ~50M |
| Language Model | Phi-3-mini-128k-instruct | ~3.8B (LoRA: ~35M trainable) |

## Quick Start

### Installation

```bash
pip install -e ".[eval,dev]"
```

### Single-Image Inference

```python
from agri_perceiver.inference.predictor import AgriPredictor

predictor = AgriPredictor("checkpoints/specialist_e3.pt")
report = predictor.predict("path/to/leaf.jpg")
print(report)
```

### CLI

```bash
# Inference
agri-predict --image leaf.jpg --checkpoint checkpoints/specialist_e3.pt

# Evaluation
agri-eval --predictions preds.jsonl --ground-truth gt.jsonl --output results.json
```

### Training

```bash
# Stage 1 — Alignment (bridge only, both backbones frozen)
python -m agri_perceiver.training.train_stage1 \
    --data data/alignment/generic_alignment.jsonl \
    --data-root /path/to/images \
    --output checkpoints/stage1_connector_weights.pt

# Stage 2 — Specialization (bridge + LoRA)
python -m agri_perceiver.training.train_stage2 \
    --data final_train_canonical.jsonl \
    --data-root canonical_dataset/processed_images/ \
    --bridge-weights checkpoints/stage1_connector_weights.pt \
    --output checkpoints/specialist_e{epoch}.pt
```

## Repository Structure

```
agri-perceiver/
├── pyproject.toml                     # Package config, deps, CLI entry points
├── configs/
│   ├── model.yaml                     # Architecture hyperparameters
│   ├── stage1_alignment.yaml          # Stage 1 training config
│   ├── stage2_specialization.yaml     # Stage 2 training config
│   └── eval.yaml                      # Evaluation config
├── src/agri_perceiver/
│   ├── model/                         # Core architecture
│   │   ├── agri_vlm.py               # Main VLM (encode → splice → generate)
│   │   ├── perceiver_resampler.py     # Perceiver bridge (cross/self-attn + GLU-FFN)
│   │   ├── vision_projector.py        # MLP dimension projector
│   │   ├── tile_embeddings.py         # Learned spatial tile positions
│   │   └── backbones.py               # Vision/LLM loading utilities
│   ├── data/                          # Datasets and collation
│   ├── training/                      # Stage 1 & 2 training scripts
│   ├── inference/
│   │   ├── predictor.py               # Single-call inference API
│   │   └── schema.py                  # Pydantic diagnostic report schema
│   └── evaluation/
│       ├── metrics.py                 # 9 evaluation metrics + composite score
│       ├── baselines.py               # Gemma-3, LLaVA-NeXT, InternVL2 runners
│       ├── judge.py                   # LLM-as-judge multi-consensus evaluator
│       ├── run_eval.py                # Master evaluation runner
│       └── report.py                  # Markdown report generation
├── checkpoints/                       # Model weights (git-ignored)
├── data_pipeline/                     # Data labeling scripts (Gemma-3 pathologist)
├── docs/                              # Architecture, training, evaluation docs + archived logs
├── tests/                             # pytest suite (16 tests)
└── _legacy/                           # Original pre-refactor code (git-ignored)
```

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Structural | JSON validity, schema compliance |
| Classification | Type F1 (macro/weighted), diagnosis exact/fuzzy match |
| Regression | Severity MAE, RMSE, Pearson r |
| Calibration | Expected Calibration Error (ECE) |
| Semantic | BERTScore for symptoms, reasoning, actions |
| Expert | LLM-as-judge (5 axes × multi-judge consensus) |
| Aggregate | Weighted composite score |

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full architecture deep dive with tensor shape flow
- [docs/TRAINING.md](docs/TRAINING.md) — Training guide, LoRA config, hardware requirements
- [docs/EVALUATION.md](docs/EVALUATION.md) — Metric definitions, baselines, judge protocol
- [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) — Data labeling pipeline (Gemma-3 → refinement)

## License

Apache 2.0


