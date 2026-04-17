# Training Guide

## Two-Stage Training Paradigm

AgriPerceiver follows the established VLM two-stage curriculum:

| | Stage 1: Alignment | Stage 2: Specialization |
|---|---|---|
| **Objective** | Align visual features to LLM embedding space | Teach structured agricultural diagnostics |
| **Data** | Image–caption pairs | Image–JSON diagnostic report pairs |
| **Trainable** | Bridge only (~50M params) | Bridge + LoRA (~85M params) |
| **Frozen** | SigLIP + Phi-3 (both) | SigLIP only |
| **Loss** | Causal LM (captioning) | Weighted cross-entropy (JSON generation) |
| **Epochs** | 2 | 3 |
| **LR** | 1e-4 | 1e-4 |
| **Batch size** | 16 | 8 (effective 16 with grad accum 2) |

## Stage 1 — Alignment

Trains the perception bridge (TileEmbeddings → VisionProjector → PerceiverResampler) to project visual features into the LLM's token space. Both backbones are frozen.

```bash
python -m agri_perceiver.training.train_stage1 \
    --data data/alignment/generic_alignment.jsonl \
    --data-root /path/to/images \
    --output checkpoints/stage1_connector_weights.pt \
    --epochs 2 \
    --batch-size 16 \
    --lr 1e-4
```

**Output:** `stage1_connector_weights.pt` (~745 MB) containing:
- `tile_embed` state dict
- `projector` state dict
- `perceiver` state dict

## Stage 2 — Specialization

Loads Stage 1 bridge weights, applies LoRA adapters to Phi-3, and trains on structured JSON diagnostic targets with per-sample weighting from Gemma-3 confidence scores.

```bash
python -m agri_perceiver.training.train_stage2 \
    --data final_train_canonical.jsonl \
    --data-root canonical_dataset/processed_images/ \
    --bridge-weights checkpoints/stage1_connector_weights.pt \
    --output checkpoints/specialist_e{epoch}.pt \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 2
```

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha | 64 |
| Target modules | `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `up_proj` |
| Dropout | 0.1 |
| Effective scaling | α/r = 2.0 |

### Weighted Loss

The cross-entropy loss is computed per-sample with reduction='none', then weighted by `sample_weight` (derived from Gemma-3 math_confidence). This curriculum effect emphasizes high-confidence labels while retaining harder samples at reduced influence.

```
loss_per_sample = CE(shift_logits, shift_labels).mean(dim=seq)
weighted_loss = (loss_per_sample * sample_weight).mean() / grad_accum
```

## Checkpoints

| File | Size | Contents |
|------|------|----------|
| `stage1_connector_weights.pt` | 745 MB | Bridge only (tile_embed + projector + perceiver) |
| `specialist_e{1,2,3}.pt` | ~9.8 GB | Full model state dict (all params including frozen backbones + LoRA) |

## Hardware Requirements

- **Stage 1**: ~20 GB VRAM (SigLIP + Phi-3 in bf16 + bridge gradients)
- **Stage 2**: ~35 GB VRAM (full model + LoRA + gradient accumulation)
- **Original training**: NVIDIA Quadro RTX 8000 (48 GB) and H200

## Key Implementation Details

1. **`use_cache=False`** in Phi-3 forward pass to avoid DynamicCache errors during splice-and-forward
2. **Explicit bf16 casting** throughout to prevent dtype mismatches
3. **`num_workers=0`** in DataLoader to avoid shared memory bus errors
4. **Tokenizer expansion**: 3 special tokens (`<image>`, `<image_start>`, `<image_end>`) added before embedding resize
