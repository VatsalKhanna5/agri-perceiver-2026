# AgriPerceiver VLM — Complete Architecture & Repository Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Visual Pipeline — Detailed Component Breakdown](#3-visual-pipeline--detailed-component-breakdown)
   - 3.1 [AnyRes Tiling Strategy](#31-anyres-tiling-strategy)
   - 3.2 [SigLIP Vision Encoder](#32-siglip-vision-encoder)
   - 3.3 [Tile Embeddings (Spatial Grounding)](#33-tile-embeddings-spatial-grounding)
   - 3.4 [Vision Projector](#34-vision-projector)
   - 3.5 [Perceiver Resampler](#35-perceiver-resampler)
4. [Language Model — Phi-3 Mini 128K Instruct](#4-language-model--phi-3-mini-128k-instruct)
5. [Multimodal Fusion — The Splice-and-Forward Mechanism](#5-multimodal-fusion--the-splice-and-forward-mechanism)
6. [Training Strategy](#6-training-strategy)
   - 6.1 [Stage 1 — Visual-Language Alignment (Pretraining)](#61-stage-1--visual-language-alignment-pretraining)
   - 6.2 [Stage 2 — Agricultural Specialization (Fine-tuning)](#62-stage-2--agricultural-specialization-fine-tuning)
7. [Datasets](#7-datasets)
   - 7.1 [Alignment Dataset (Stage 1)](#71-alignment-dataset-stage-1)
   - 7.2 [AgriVLM Dataset (Stage 2)](#72-agrivlm-dataset-stage-2)
   - 7.3 [Collation Functions](#73-collation-functions)
8. [Inference Pipeline](#8-inference-pipeline)
9. [HuggingFace Export — KisaanMithraa-Drishti](#9-huggingface-export--kisaanmithraa-drishti)
10. [Repository Structure](#10-repository-structure)
11. [Tensor Shape Flow — End-to-End](#11-tensor-shape-flow--end-to-end)
12. [Configuration & Environment](#12-configuration--environment)
13. [Monitoring & Utilities](#13-monitoring--utilities)
14. [Design Decisions & Rationale](#14-design-decisions--rationale)

---

## 1. Project Overview

**AgriPerceiver VLM** (also published as **KisaanMithraa-Drishti**) is a domain-specific Vision-Language Model designed for **agricultural pathology analysis**. Given a photograph of a plant leaf, it generates structured JSON diagnostic reports covering:

- Visual symptom description (venation, margins, lesions, color)
- Suspected pathogen or deficiency identification
- Pathological reasoning

The model follows the established two-stage VLM training paradigm (pioneered by LLaVA and similar architectures), adapted with a custom Perceiver-based visual bridge for high-resolution agricultural imagery.

### Core Design Philosophy

| Principle | Implementation |
|-----------|----------------|
| **High-resolution fidelity** | AnyRes 5-tile strategy preserving local detail |
| **Efficient visual compression** | Perceiver Resampler compresses 3,645 visual tokens → 128 latents |
| **Domain specialization** | Two-stage curriculum: alignment → agricultural fine-tuning |
| **Structured output** | Model trained to produce JSON-formatted pathology reports |
| **Parameter efficiency** | LoRA adapters on Phi-3 during Stage 2 (only ~2% trainable params) |

### Backbone Components

| Component | Model | Parameters | Role |
|-----------|-------|------------|------|
| Vision Encoder | SigLIP-SO400M-patch14-384 | ~400M | Patch-level visual feature extraction |
| Language Model | Phi-3-mini-128k-instruct | ~3.8B | Text generation and reasoning |
| Visual Bridge | Custom (Perceiver + Projector + TileEmbed) | ~50M | Cross-modal alignment |

---

## 2. High-Level Architecture

```
                    Input Image (arbitrary resolution)
                            │
                    ┌───────▼────────┐
                    │  AnyRes Tiling  │
                    │ (4 quadrants +  │
                    │  1 global view) │
                    └───────┬────────┘
                            │ [B, 5, 3, 384, 384]
                    ┌───────▼────────┐
                    │  SigLIP ViT    │  (FROZEN)
                    │  SO400M/14     │
                    └───────┬────────┘
                            │ [B, 5, 729, 1152]
                    ┌───────▼────────┐
                    │ Tile Embeddings │  (TRAINABLE)
                    │ Spatial Ground. │
                    └───────┬────────┘
                            │ [B, 5, 729, 1152]
                            │ reshape → [B, 3645, 1152]
                    ┌───────▼────────┐
                    │ Vision Projector│  (TRAINABLE)
                    │ 1152 → 3072    │
                    └───────┬────────┘
                            │ [B, 3645, 3072]
                    ┌───────▼────────┐
                    │   Perceiver    │  (TRAINABLE)
                    │   Resampler    │
                    │ 3645 → 128     │
                    └───────┬────────┘
                            │ [B, 128, 3072]
                            │
              ┌─────────────▼──────────────┐
              │    Splice-and-Forward       │
              │                            │
              │  [text_before] +           │
              │  [<image_start>] +         │
              │  [128 visual latents] +    │
              │  [<image_end>] +           │
              │  [text_after]              │
              └─────────────┬──────────────┘
                            │ [B, S, 3072] (hybrid embeddings)
                    ┌───────▼────────┐
                    │    Phi-3 Mini   │  Stage 1: FROZEN
                    │    128K Inst.   │  Stage 2: LoRA
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  JSON Report   │
                    │  Generation    │
                    └────────────────┘
```

---

## 3. Visual Pipeline — Detailed Component Breakdown

### 3.1 AnyRes Tiling Strategy

**File:** Implemented inline in datasets (`data/datasets/alignment_dataset.py`, `data/datasets/agri_vlm_dataset.py`, `inference_test.py`)

The model uses an **AnyRes (Any Resolution)** tiling approach to preserve spatial detail from high-resolution agricultural photographs. Every input image, regardless of its native resolution, is decomposed into **5 tiles**:

```
Original Image (H × W)
┌─────────┬─────────┐
│  Tile 0  │  Tile 1  │   Top-Left, Top-Right quadrants
│  (TL)    │  (TR)    │
├─────────┼─────────┤
│  Tile 2  │  Tile 3  │   Bottom-Left, Bottom-Right quadrants
│  (BL)    │  (BR)    │
└─────────┴─────────┘
        +
┌─────────────────────┐
│      Tile 4          │   Global view (full image resized)
│      (Global)        │
└─────────────────────┘
```

**Tiling Algorithm:**
```python
h, w = img.shape[:2]
mid_h, mid_w = h // 2, w // 2

tiles = [
    img[:mid_h, :mid_w],       # Tile 0: Top-Left
    img[:mid_h, mid_w:],       # Tile 1: Top-Right
    img[mid_h:, :mid_w],       # Tile 2: Bottom-Left
    img[mid_h:, mid_w:],       # Tile 3: Bottom-Right
    cv2.resize(img, (w, h))    # Tile 4: Global (identity resize)
]

# Each tile → resize to 384×384 → normalize to [0, 1] → [3, 384, 384]
```

**Rationale:** Agricultural pathology requires both macro-level context (overall leaf shape, global color patterns) and micro-level detail (individual lesion morphology, fungal structures). The 4 quadrants preserve local detail at 2× effective resolution, while the global view maintains holistic context.

**Output shape:** `[B, 5, 3, 384, 384]`

---

### 3.2 SigLIP Vision Encoder

**File:** [models/vision/siglip_wrapper.py](models/vision/siglip_wrapper.py)

**Model:** `google/siglip-so400m-patch14-384`

SigLIP (Sigmoid Loss for Language-Image Pre-training) is used as the frozen visual backbone. Key specifications:

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-SO400M |
| Patch size | 14×14 pixels |
| Input resolution | 384×384 |
| Patches per tile | 27×27 = 729 (384/14 = 27.4, rounded) |
| Hidden dimension | 1152 |
| Pre-training | Sigmoid contrastive loss on web-scale data |

**Processing flow:**
1. All 5 tiles are flattened to `[B*5, 3, 384, 384]` for batch processing
2. SigLIP's ViT extracts patch tokens: `[B*5, 729, 1152]`
3. Reshaped back: `[B, 5, 729, 1152]`

The SigLIP wrapper (`SigLIPWrapper`) provides a clean interface, though during actual training/inference the raw `AutoModel` is used directly via `model.siglip.vision_model()`.

**Why SigLIP over CLIP?** SigLIP uses sigmoid-based loss (per-pair binary classification) instead of CLIP's softmax-based contrastive loss. This yields better calibrated features and improved performance on fine-grained visual tasks — critical for distinguishing subtle agricultural symptoms.

**Status:** Always **FROZEN** (both Stage 1 and Stage 2). Only its output features are used.

---

### 3.3 Tile Embeddings (Spatial Grounding)

**File:** [models/spatial/tile_embeddings.py](models/spatial/tile_embeddings.py)

```python
class TileEmbeddings(nn.Module):
    def __init__(self, dim=1152, num_tiles=5, dropout=0.0):
        self.tile_embeddings = nn.Parameter(torch.randn(num_tiles, dim) * 0.02)
```

**Purpose:** After SigLIP encodes all 5 tiles identically, the model has no way to distinguish *which spatial region* each tile came from. Tile Embeddings solve this by adding a **learned positional embedding per tile**.

**Mechanism:**
- 5 learnable embedding vectors of dimension 1152 (one per tile position)
- Broadcast-added to every patch token within each tile
- Standard deviation init of 0.02 (small perturbation, not disruptive)

```
Input:  [B, 5, 729, 1152]
         │
    + tile_embeddings[i] for each tile i
         │
Output: [B, 5, 729, 1152]  (now spatially grounded)
```

**Analogy:** This is analogous to positional embeddings in standard Transformers, but operates at the *tile level* rather than the token level. Tile 0 (top-left) gets a different learned bias than Tile 4 (global), allowing downstream modules to understand spatial layout.

**Trainable parameters:** 5 × 1152 = **5,760 parameters**

---

### 3.4 Vision Projector

**File:** [models/projector/vision_projector.py](models/projector/vision_projector.py)

```python
class VisionProjector(nn.Module):
    def __init__(self, in_dim=1152, out_dim=3072, dropout=0.1):
        self.net = nn.Sequential(
            nn.Linear(1152, 3072),
            nn.GELU(),
            nn.Linear(3072, 3072),
            nn.Dropout(0.1),
        )
```

**Purpose:** Bridges the dimensional gap between SigLIP's visual feature space (dim=1152) and Phi-3's language embedding space (dim=3072).

**Architecture:** Two-layer MLP with GELU activation:
```
[B, 3645, 1152] → Linear(1152→3072) → GELU → Linear(3072→3072) → Dropout(0.1) → [B, 3645, 3072]
```

**Note:** Before projection, all tile features are flattened:
```
[B, 5, 729, 1152] → reshape → [B, 3645, 1152]
```

This produces 3,645 visual tokens at Phi-3's native dimension.

**Initialization:** Xavier uniform for weights, zeros for biases — ensures stable gradient flow at initialization.

**Trainable parameters:** (1152 × 3072 + 3072) + (3072 × 3072 + 3072) = ~12.8M parameters

---

### 3.5 Perceiver Resampler

**File:** [models/perceiver/perceiver_resampler.py](models/perceiver/perceiver_resampler.py)

The Perceiver Resampler is the most architecturally significant custom component. It compresses the 3,645 projected visual tokens down to a fixed set of **128 learned latent vectors**, acting as an information bottleneck.

**Architecture:**

```python
class PerceiverResampler(nn.Module):
    def __init__(self, dim=3072, num_latents=128, depth=2, heads=24):
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverBlock(dim, heads) for _ in range(depth)])
        self.norm = RMSNorm(dim)
```

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| `dim` | 3072 | Matches Phi-3 hidden dimension |
| `num_latents` | 128 | Number of output visual tokens |
| `depth` | 2 | Number of Perceiver blocks |
| `heads` | 24 | Multi-head attention heads |

**Each PerceiverBlock contains:**

1. **Cross-Attention** (latents attend to visual tokens):
   ```
   Q = latents [B, 128, 3072]
   K, V = visual_tokens [B, 3645, 3072]
   Output: [B, 128, 3072]
   ```
   - Custom implementation with 24 heads, scaled dot-product attention
   - Head dimension = 3072/24 = 128

2. **Self-Attention** (latents attend to each other):
   ```
   Q = K = V = latents [B, 128, 3072]
   Output: [B, 128, 3072]
   ```
   - Uses PyTorch's `nn.MultiheadAttention`

3. **Feed-Forward Network** (MLP with GLU gating):
   ```
   latents → Linear(3072 → 24576) → GLU → Linear(12288 → 3072) → latents
   ```
   - GLU (Gated Linear Unit) splits the expanded features and applies element-wise gating
   - MLP ratio = 4, so expansion = 3072 × 4 × 2 = 24,576 (×2 for GLU split)

**All three sub-layers use:**
- Pre-norm with **RMSNorm** (Root Mean Square Layer Normalization)
- Residual connections

**Compression ratio:** 3,645 → 128 tokens = **28.5× compression**

This is critical for efficiency: feeding 3,645 tokens directly into Phi-3 would be extremely expensive (quadratic attention cost). The Perceiver distills the visual information into a fixed-size representation.

**Final RMSNorm:** Applied to the output latents for stable downstream processing.

**RMSNorm implementation:**
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x
```

RMSNorm is preferred over LayerNorm in modern LLM architectures (used by Llama, Phi-3) because it's computationally cheaper (no mean subtraction) and empirically works just as well.

**Trainable parameters:** ~75M parameters (dominant component of the visual bridge)

---

## 4. Language Model — Phi-3 Mini 128K Instruct

**Model:** `microsoft/Phi-3-mini-128k-instruct`

| Specification | Value |
|---------------|-------|
| Parameters | 3.8B |
| Hidden dimension | 3072 |
| Layers | 32 |
| Attention heads | 32 |
| Context length | 128K tokens |
| Vocabulary | 32,000 + 3 special tokens |
| Precision | bfloat16 |

**Special tokens added:**
| Token | ID | Purpose |
|-------|----|---------|
| `<image>` | 32000 | Placeholder in input text (replaced by visual latents) |
| `<image_start>` | 32001 | Boundary marker before visual tokens |
| `<image_end>` | 32002 | Boundary marker after visual tokens |

The tokenizer's pad token is set to the EOS token (standard for Phi-3).

**Chat template format (Phi-3 Instruct):**
```
<|user|>
<image>
[Question/instruction text]<|assistant|>
[Model response]
```

### LoRA Configuration (Stage 2)

**File:** [models/llm/phi3_lora.py](models/llm/phi3_lora.py)

Two LoRA configurations are defined:

**Stage 2 basic (`train_stage2.py`):**
```python
LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "down_proj", "up_proj"],
    lora_dropout=0.05
)
```

**Stage 2 specialization (`train_stage2_specialization.py`):**
```python
LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "up_proj"],
    lora_dropout=0.1
)
```

| Parameter | Basic | Specialization | Notes |
|-----------|-------|----------------|-------|
| Rank (r) | 16 | 32 | Higher rank = more capacity |
| Alpha | 32 | 64 | Scaling factor (alpha/r = effective scale) |
| Target modules | 4 | 5 | Specialization adds `gate_up_proj` |
| Dropout | 0.05 | 0.1 | Higher regularization for specialization |

Phi-3 uses fused `qkv_proj` (combined Q/K/V projection), so a single LoRA adapter modifies all three attention matrices simultaneously.

---

## 5. Multimodal Fusion — The Splice-and-Forward Mechanism

**File:** [models/agri_vlm.py](models/agri_vlm.py) — `splice_and_forward()` method

This is the core mechanism that interleaves visual and textual information. Rather than using cross-attention between modalities (as in Flamingo), AgriPerceiver uses **early fusion via embedding splicing** (as in LLaVA).

### Step-by-Step Process

**Input:**
- `input_ids`: `[B, S]` — tokenized text containing an `<image>` placeholder
- `visual_latents`: `[B, 128, 3072]` — compressed visual features from the Perceiver
- `labels` (optional): `[B, S]` — training targets

**Algorithm (per batch element):**

1. **Locate the `<image>` placeholder** in the token sequence
2. **Split the text** into `before_tokens` and `after_tokens` around the placeholder
3. **Embed all text portions** through Phi-3's embedding layer
4. **Construct the hybrid sequence:**

```
[before_embed] + [<image_start>_embed] + [128 visual latents] + [<image_end>_embed] + [after_embed]
```

5. **Construct matching labels:**

```
[labels_before] + [-100] × (1 + 128 + 1) + [labels_after]
```

The `-100` values ensure that the cross-entropy loss **ignores** the visual token positions (the model shouldn't learn to "predict" visual tokens as text).

6. **Pad sequences** across the batch (using `pad_sequence`)
7. **Forward through Phi-3** with `inputs_embeds` (bypassing the embedding layer since we've already constructed the embeddings)

**Critical implementation detail:**
```python
outputs = self.phi3(
    inputs_embeds=new_embeds,
    attention_mask=new_masks,
    use_cache=False  # Avoids DynamicCache AttributeError in Phi-3
)
```

The `use_cache=False` flag is necessary because Phi-3's KV-cache implementation (`DynamicCache`) has compatibility issues when using `inputs_embeds` instead of `input_ids`.

### Sequence Layout Example

For the prompt: `"<|user|>\n<image>\nDescribe this leaf.<|assistant|>\n"`

```
Position: 0    1    2    3     4      5      ...    132   133    134   135   ...
Token:    <|   user  |>  \n  <img_s> [v_0]  ...  [v_127] <img_e> \n    Desc  ...
Label:    -100 -100 -100 -100 -100   -100   ...  -100    -100    -100  -100  ...
          ▲ prompt masked ▲   ▲ visual tokens masked ▲    ▲ prompt continued ▲
```

During training, only the assistant's response tokens contribute to the loss.

---

## 6. Training Strategy

The model follows a **two-stage curriculum** that progressively trains different components:

```
┌──────────────────────────────────────────────────┐
│              STAGE 1: ALIGNMENT                   │
│                                                    │
│  Goal: Teach the visual bridge to produce          │
│        embeddings that Phi-3 can understand        │
│                                                    │
│  Trainable: TileEmbed + Projector + Perceiver     │
│  Frozen:    SigLIP + Phi-3                         │
│  Data:      Generic image-caption pairs            │
│  Task:      "Describe this agricultural sample."   │
│  Loss:      Standard Causal LM (cross-entropy)     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│          STAGE 2: SPECIALIZATION                  │
│                                                    │
│  Goal: Fine-tune the entire system for             │
│        agricultural pathology JSON reports          │
│                                                    │
│  Trainable: TileEmbed + Projector + Perceiver     │
│             + Phi-3 LoRA adapters                  │
│  Frozen:    SigLIP + Phi-3 base weights            │
│  Data:      Agricultural leaf images + JSON reports│
│  Task:      "Provide a structured lab report"      │
│  Loss:      Weighted Causal LM (cross-entropy)     │
└──────────────────────────────────────────────────┘
```

---

### 6.1 Stage 1 — Visual-Language Alignment (Pretraining)

**Primary Script:** [training/train_stage1_alignment.py](training/train_stage1_alignment.py)  
**Alternative Script:** [training/train_stage1.py](training/train_stage1.py) (uses Accelerate for multi-GPU)

**Objective:** Train the visual bridge (TileEmbeddings + VisionProjector + PerceiverResampler) to produce visual embeddings that are semantically meaningful in Phi-3's embedding space.

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Epochs | 2 |
| Batch size | 16 |
| Precision | bfloat16 |
| Workers | 0 (avoids shared memory bus errors) |

**Frozen Components:**
- SigLIP (entire model): `model.siglip.requires_grad_(False)`
- Phi-3 (entire model): `model.phi3.requires_grad_(False)`

**Trainable Components** (only the visual bridge):
- `model.tile_embed` — 5,760 params
- `model.projector` — ~12.8M params
- `model.perceiver` — ~75M params

**Data:** `AlignmentDataset` — image-caption pairs from `data/alignment/generic_alignment.jsonl`

**Prompt template:**
```
<|user|>
<image>
Describe this agricultural sample.<|assistant|>
[caption text]<eos>
```

**Label masking:** Prompt tokens are masked with `-100`. Only the caption text (assistant response) contributes to the cross-entropy loss.

**Output:** `stage1_connector_weights.pt` — a dictionary containing:
```python
{
    "tile_embed": model.tile_embed.state_dict(),
    "projector": model.projector.state_dict(),
    "perceiver": model.perceiver.state_dict(),
}
```

**Multi-GPU variant** (`train_stage1.py`):
- Uses HuggingFace `Accelerator` with `mixed_precision="bf16"`
- Higher learning rate: 1e-3
- Uses `AgriDataset` with pre-tiled `.pt` files instead of raw images

---

### 6.2 Stage 2 — Agricultural Specialization (Fine-tuning)

**Primary Script:** [training/train_stage2_specialization.py](training/train_stage2_specialization.py)  
**Alternative Script:** [training/train_stage2.py](training/train_stage2.py) (simpler version)

**Objective:** Fine-tune the entire model (via LoRA on Phi-3) to produce domain-specific structured JSON diagnostic reports.

**Training Configuration (Specialization):**

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch = 16) |
| Precision | bfloat16 |
| Seed | 42 |

**Trainable Components:**
- Visual bridge (TileEmbed + Projector + Perceiver) — continued training
- Phi-3 LoRA adapters (r=32, alpha=64) — newly attached
- Frozen: SigLIP backbone, Phi-3 base weights

**LoRA Target Modules:**
- `qkv_proj` — fused query/key/value projection
- `o_proj` — output projection
- `gate_up_proj` — gated MLP upward projection
- `down_proj` — MLP downward projection
- `up_proj` — MLP upward projection

**Initialization:** Stage 1 connector weights are loaded first:
```python
st1_weights = torch.load("stage1_connector_weights.pt", map_location=DEVICE)
model.tile_embed.load_state_dict(st1_weights["tile_embed"])
model.projector.load_state_dict(st1_weights["projector"])
model.perceiver.load_state_dict(st1_weights["perceiver"])
```

**Data:** `AgriVLM_Dataset` — leaf images paired with canonical JSON reports from `final_train_canonical.jsonl`

**Prompt template:**
```
<|user|>
<image>
You are an agricultural pathology AI. Analyze this leaf and provide a structured lab report in JSON.
<|assistant|>
{"visual_summary": {...}, "pathogen_id": {...}, ...}<eos>
```

**Weighted Loss Computation:**

The specialization script implements **sample-weighted cross-entropy loss**:

```python
# Standard shifted causal LM loss (predict next token)
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = correct_labels[..., 1:].contiguous()

# Per-token loss
loss_fct = nn.CrossEntropyLoss(reduction='none')
loss = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))

# Reshape to [B, S-1], mean across sequence, apply per-sample weights
loss = loss.view(B, -1).mean(dim=1)
weighted_loss = (loss * weights).mean() / GRAD_ACCUM
```

Each sample has a `sample_weight` field (from the dataset), allowing certain training examples to have higher influence on the loss (e.g., rare disease classes can be upweighted).

**Checkpointing:** Full model state dict saved per epoch:
- `agri_perceiver_specialist_e1.pt`
- `agri_perceiver_specialist_e2.pt`
- `agri_perceiver_specialist_e3.pt`

**Important difference in `forward()` return:**

The specialization script uses the **corrected labels** returned by `splice_and_forward()`:

```python
outputs, correct_labels = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    labels=labels
)
```

After splicing, the label sequence is longer (128 visual tokens + 2 boundary tokens inserted). The model returns these adjusted labels for correct loss computation.

---

## 7. Datasets

### 7.1 Alignment Dataset (Stage 1)

**File:** [data/datasets/alignment_dataset.py](data/datasets/alignment_dataset.py)

**Data source:** `data/alignment/generic_alignment.jsonl`

**JSONL format:**
```json
{
    "image": "canonical_dataset/processed_images/agri_000000.jpg",
    "caption": "A plant leaf showing [detailed description]..."
}
```

**Processing pipeline per sample:**
1. Load image via OpenCV (`cv2.imread`)
2. Convert BGR → RGB
3. Apply AnyRes tiling → `[5, 3, 384, 384]`
4. Construct prompt: `<|user|>\n<image>\nDescribe this agricultural sample.<|assistant|>\n`
5. Tokenize prompt and caption separately
6. Concatenate `input_ids = [prompt_ids, caption_ids]`
7. Create labels: mask prompt portion with `-100`

**Key details:**
- Images are loaded from raw files (not pre-processed tiles)
- Normalization: simple `/255.0` (no ImageNet normalization — SigLIP uses its own)
- Caption descriptions include visual features, pathogen identification, and reasoning

---

### 7.2 AgriVLM Dataset (Stage 2)

**File:** [data/datasets/agri_vlm_dataset.py](data/datasets/agri_vlm_dataset.py)

**Data source:** `final_train_canonical.jsonl`

**JSONL format:**
```json
{
    "image": "path/to/leaf_image.jpg",
    "canonical_report": {
        "visual_summary": {...},
        "pathogen_id": {...},
        ...
    },
    "sample_weight": 1.0
}
```

**Processing pipeline per sample:**
1. Load image → AnyRes tiling → `[5, 3, 384, 384]`
2. Fixed specialist prompt: `"You are an agricultural pathology AI. Analyze this leaf and provide a structured lab report in JSON."`
3. Target = `json.dumps(sample["canonical_report"])` + EOS token
4. Tokenize full text, mask prompt portion with `-100`
5. Return `sample_weight` for weighted loss

**Robustness:**
- If an image fails to load (`cv2.imread` returns `None`), the dataset falls back to the next sample: `return self.__getitem__((idx + 1) % len(self.samples))`
- Max sequence length: 2048 tokens

---

### 7.3 Collation Functions

**File:** [training/collate_fn.py](training/collate_fn.py)

Two collation functions handle batching:

**`alignment_collate_fn`** (Stage 1):
- Stacks pixel values: `[B, 5, 3, 384, 384]`
- Pads `input_ids` with 0
- Pads `attention_mask` with 0
- Pads `labels` with -100

**`agri_collate_fn`** (Stage 2):
- Same as above, plus:
- Dynamically generates `attention_mask` from `input_ids != 0`
- Stacks `sample_weight` tensors

---

## 8. Inference Pipeline

### Stage 1 Inference Test

**File:** [inference_test.py](inference_test.py)

Tests Stage 1 alignment weights with manual greedy decoding:

1. Load Phi-3 + SigLIP (no LoRA)
2. Load `stage1_connector_weights.pt`
3. Tile input image → `[1, 5, 3, 384, 384]`
4. Encode: `visual_latents = model.encode_images(pixel_values)`
5. Manual token-by-token generation (150 steps max):
   ```python
   for _ in range(150):
       outputs = model.splice_and_forward(input_ids, curr_mask, visual_latents)
       next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
       # append token, continue...
   ```

### Stage 2 Specialist Testing

**File:** [test.py](test.py) — Tests against known dataset samples with ground truth  
**File:** [test2.py](test2.py) — Tests against arbitrary images in a folder

Both scripts:
1. Load Phi-3 + SigLIP + LoRA (must re-apply exact LoRA config)
2. Load `agri_perceiver_specialist_e3.pt` (full state dict)
3. Use manual greedy decoding (256–350 tokens max)
4. Print generated JSON reports

**test.py** additionally:
- Randomly samples from the training JSONL
- Prints ground truth JSON for comparison

**test2.py** additionally:
- Uses robust image loading via `pathlib` and `np.frombuffer` (handles edge cases like broken symlinks)
- Processes all valid images in a target directory
- Uses `attn_implementation="eager"` for Phi-3

---

## 9. HuggingFace Export — KisaanMithraa-Drishti

**Directory:** `KisaanMithraa-Drishti/`

This folder contains a HuggingFace-compatible model export for deployment/distribution.

### Structure

```
KisaanMithraa-Drishti/
├── config.json                    # Model architecture config
├── code/
│   ├── modeling_kisaanmitra.py    # HF-compatible model wrapper
│   └── models/                    # Copy of the core model code
├── model/
│   ├── language_model/            # Merged Phi-3 + LoRA weights
│   │   ├── model.safetensors      # Phi-3 weights (safetensors format)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   └── tokenizer files...
│   └── vision_encoder/            # SigLIP weights
│       ├── model.safetensors
│       └── config.json
└── tokenizer/                     # Tokenizer with special tokens
    ├── tokenizer.json
    └── tokenizer_config.json
```

### `config.json`

```json
{
    "model_type": "kisaanmithraa_drishti",
    "vision_config": {
        "backbone": "google/siglip-so400m-patch14-384",
        "image_size": 384,
        "patch_size": 14,
        "num_tiles": 5
    },
    "bridge_config": {
        "projector_type": "linear",
        "perceiver_latents": 128,
        "vision_dim": 1152,
        "language_dim": 3072
    }
}
```

### `modeling_kisaanmitra.py`

The `KisaanMitraVLM` class wraps the model for HuggingFace distribution:
- `from_pretrained(model_path)` — loads vision tower, language model, and bridge weights from the export directory
- `generate_report(pixel_values, tokenizer, prompt)` — high-level inference API (stub/incomplete)

**Note:** The LoRA adapters have been **merged** into the base Phi-3 weights in the export (the model directory contains the merged `model.safetensors` without separate LoRA adapters).

---

## 10. Repository Structure

```
agri-perceiver/
│
├── models/                          # Core model architecture
│   ├── agri_vlm.py                 # ★ Main VLM class (AgriPerceiverVLM)
│   ├── vision/
│   │   └── siglip_wrapper.py       # SigLIP encoder wrapper
│   ├── spatial/
│   │   └── tile_embeddings.py      # Learned tile positional embeddings
│   ├── projector/
│   │   └── vision_projector.py     # 1152→3072 MLP projector
│   ├── perceiver/
│   │   └── perceiver_resampler.py  # Perceiver Resampler (3645→128 tokens)
│   └── llm/
│       └── phi3_lora.py            # Phi-3 base & LoRA loading utilities
│
├── training/                        # Training scripts
│   ├── train_stage1.py             # Stage 1 with Accelerate (multi-GPU)
│   ├── train_stage1_alignment.py   # ★ Stage 1 alignment (single-GPU, production)
│   ├── train_stage2.py             # Stage 2 basic fine-tuning
│   ├── train_stage2_specialization.py # ★ Stage 2 specialization (production)
│   └── collate_fn.py              # Batch collation functions
│
├── data/                            # Data loading & datasets
│   ├── dataset_loader.py           # AgriDataset (pre-tiled .pt files)
│   ├── datasets/
│   │   ├── alignment_dataset.py    # Stage 1 image-caption dataset
│   │   └── agri_vlm_dataset.py     # Stage 2 image-JSON report dataset
│   └── alignment/
│       └── generic_alignment.jsonl # Stage 1 training data
│
├── inference/                       # Inference modules (stubs)
│   ├── api_server.py               # FastAPI server (empty)
│   ├── generate_json.py            # JSON generation (empty)
│   ├── outlines_schema.py          # Constrained decoding schema (empty)
│   └── postprocess.py              # Post-processing (empty)
│
├── configs/                         # Configuration files
│   └── paths.yaml                  # Data root & annotation paths
│
├── KisaanMithraa-Drishti/          # HuggingFace model export
│   ├── config.json
│   ├── code/                       # Model code for HF
│   └── model/                      # Serialized weights
│
├── monitoring/                      # GPU monitoring utilities
│   ├── gpu_logger.py               # Background GPU stats logger
│   └── gpu_usage.py                # GPU log analyzer
│
├── tests/                           # Unit tests
│   ├── test_dataset.py             # Dataset loading test
│   └── test_forward_pass.py        # Forward pass with dummy models
│
├── inference_test.py               # Stage 1 inference test
├── test.py                         # Stage 2 test (vs ground truth)
├── test2.py                        # Stage 2 test (arbitrary images)
├── setup.py                        # Package setup
│
├── stage1_connector_weights.pt     # ★ Stage 1 trained weights
├── agri_perceiver_specialist_e1.pt # Stage 2 epoch 1 checkpoint
├── agri_perceiver_specialist_e2.pt # Stage 2 epoch 2 checkpoint
└── agri_perceiver_specialist_e3.pt # ★ Stage 2 final checkpoint
```

---

## 11. Tensor Shape Flow — End-to-End

This section traces the exact tensor shapes through the entire forward pass:

```
INPUT
  image: [H, W, 3] (arbitrary resolution, e.g., 1024×768)
  prompt: "<|user|>\n<image>\nAnalyze this leaf...\n<|assistant|>\n"

TILING
  tiles: [5, H', W', 3] → resize each to [384, 384]
  pixel_values: [B, 5, 3, 384, 384]

SIGLIP ENCODING
  flatten: [B*5, 3, 384, 384]
  SigLIP ViT output: [B*5, 729, 1152]
  reshape: [B, 5, 729, 1152]

TILE EMBEDDINGS
  + tile_embed[5, 1152] broadcast
  output: [B, 5, 729, 1152]

FLATTEN TILES
  reshape: [B, 5×729, 1152] = [B, 3645, 1152]

VISION PROJECTOR
  Linear(1152→3072) + GELU + Linear(3072→3072) + Dropout
  output: [B, 3645, 3072]

PERCEIVER RESAMPLER
  latents init: [128, 3072] → expand: [B, 128, 3072]
  PerceiverBlock ×2:
    Cross-Attn(Q=[B,128,3072], KV=[B,3645,3072]) → [B,128,3072]
    Self-Attn(Q=K=V=[B,128,3072]) → [B,128,3072]
    FFN(GLU) → [B,128,3072]
  RMSNorm
  visual_latents: [B, 128, 3072]

TOKENIZATION
  prompt tokens: [B, P] (e.g., P=25)
  tokenizer → input_ids: [B, S]

SPLICE-AND-FORWARD
  before_embed: [B, pos, 3072]
  start_embed: [B, 1, 3072]
  visual_latents: [B, 128, 3072]
  end_embed: [B, 1, 3072]
  after_embed: [B, S-pos-1, 3072]
  
  spliced: [B, pos + 1 + 128 + 1 + (S-pos-1), 3072]
         = [B, S + 130, 3072]

PHI-3 FORWARD
  input: [B, S+130, 3072]
  output logits: [B, S+130, 32003]

GENERATION (inference)
  argmax over logits[:, -1, :] → next token
  repeat until EOS or max_tokens
```

---

## 12. Configuration & Environment

### `configs/paths.yaml`
```yaml
DATA_ROOT: "/home/vats/canonical_dataset/processed_images"
ANNOTATIONS: "/home/vats/final_train_canonical.jsonl"
```

### Dependencies (from README)

```
torch, torchvision, torchaudio (CUDA 12.1)
transformers, accelerate, peft, bitsandbytes
timm, einops, datasets
wandb, fastapi, uvicorn, outlines
pydantic, pillow, opencv-python, tqdm
```

### Hardware Requirements

- **Training:** The model was developed on a system with NVIDIA H200 GPU (indicated by inference_test.py comments)
- **Inference:** Any CUDA-capable GPU with ≥16GB VRAM (bfloat16 — ~8GB for model weights)
- **Precision:** bfloat16 throughout (critical — all components are explicitly cast)

---

## 13. Monitoring & Utilities

### GPU Logger

**File:** [monitoring/gpu_logger.py](monitoring/gpu_logger.py)

Background process that logs GPU statistics every 60 seconds via `nvidia-smi`:
- GPU utilization %
- Memory utilization %
- Memory used/total

**Usage:** `nohup python monitoring/gpu_logger.py &`

### GPU Usage Analyzer

**File:** [monitoring/gpu_usage.py](monitoring/gpu_usage.py)

Reads `gpu_usage.log` and reports average utilization statistics.

### Unit Tests

**[tests/test_forward_pass.py](tests/test_forward_pass.py):** Creates dummy SigLIP/Phi-3/Tokenizer models and runs a forward pass to verify the architecture connects correctly without loading real weights.

**[tests/test_dataset.py](tests/test_dataset.py):** Verifies dataset loading and collation produces expected tensor shapes.

---

## 14. Design Decisions & Rationale

### Why Perceiver Resampler instead of simple pooling?

Simple pooling (e.g., average pooling across patches) destroys spatial information. The Perceiver's learned latent queries can selectively attend to the most informative visual tokens, preserving fine-grained details about lesion boundaries, color gradients, and texture patterns that are critical for pathology diagnosis.

### Why 128 latents?

This is a balance between information preservation and computational efficiency:
- 128 latents at dim=3072 adds ~128 tokens to Phi-3's sequence — comparable to a short text paragraph
- The 28.5× compression from 3,645 tokens is aggressive but sufficient because agricultural pathology features are spatially redundant (a leaf has repetitive texture patterns)

### Why SigLIP over CLIP?

SigLIP's sigmoid-based training produces features with better fine-grained discrimination. CLIP's softmax contrastive loss can collapse subtle visual differences when the batch contains similar images — a common scenario in agricultural datasets where many images are "green leaves."

### Why two-stage training?

- **Stage 1** (alignment) trains only the bridge with a simple captioning objective. This is a much easier optimization problem — the bridge just needs to project visual features into a space Phi-3 understands
- **Stage 2** (specialization) then fine-tunes the LLM with LoRA. If LoRA were applied from the start, the bridge and LLM would be co-adapting to a complex structured output task, making optimization unstable

### Why manual greedy decoding in inference?

The `splice_and_forward` mechanism constructs custom `inputs_embeds` that bypass HuggingFace's `generate()` API. The visual latents need to be injected into the embedding sequence, which isn't natively supported by `model.generate()`. Hence, a manual decoding loop is used.

### Why `use_cache=False`?

Phi-3's `DynamicCache` implementation throws `AttributeError` when using `inputs_embeds` (it expects `input_ids` for cache key management). Disabling KV-caching is a correctness fix at the cost of inference speed. For production, implementing a custom cache handler would resolve this.

### Why bfloat16 everywhere?

bfloat16 maintains the same dynamic range as float32 (8-bit exponent) while halving memory. This is critical for fitting the ~4B parameter model on a single GPU. The code includes multiple explicit `.to(torch.bfloat16)` calls as safety guards against dtype mismatches.

---

*This document describes the AgriPerceiver VLM architecture as implemented in the repository. Model weights (`.pt` and `.safetensors` files) are not included in this documentation.*
