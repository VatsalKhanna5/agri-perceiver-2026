<div align="center">

<img src="assets/banner.png" alt="AgriPerceiver" style="border-radius: 14px; width: 100%; max-width: 960px;" />

# AgriPerceiver

### A Parameter-Efficient Vision-Language Model for Structured Macroscopic Crop Phenotyping

*ICML 2026 FM4LS Workshop &nbsp;·&nbsp; ICA 2026 (Springer CCIS)*

<br/>

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://icml.cc/virtual/2026/72095)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](#)
[![Workshop](https://img.shields.io/badge/FM4LS%20Workshop-ICML%202026-4b8bbe.svg)](https://icml2026fm4ls.github.io/index.html)
[![ICML 2026](https://img.shields.io/badge/ICML-2026%20Seoul-0066cc.svg)](https://icml.cc/)
[![ICA 2026](https://img.shields.io/badge/ICA-2026%20IIT%20Ropar-2e7d32.svg)](https://ica26.annam.ai/)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Coming%20Soon-yellow.svg)](#)
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Coming%20Soon-yellow.svg)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

<br/>

**[Vatsal Khanna](https://github.com/VatsalKhanna5)**  &nbsp;&amp;&nbsp;  **[Davinder Singh](https://github.com/Davinder1436)**

</div>

---

## Overview

AgriPerceiver is a lightweight, domain-specialised vision-language model for structured agricultural pathology diagnosis. Given a photograph of a plant leaf, it produces a structured JSON diagnostic report covering disease identity, pathology type, severity, visual symptoms, pathological reasoning, and recommended actions.

The central contribution is a **perception bridge** built around a Perceiver Resampler, which compresses 3,645 patch-level visual tokens into 128 latents (a 28.5× reduction) while keeping learned tile-position embeddings for spatial grounding. Both backbones stay frozen. The bridge is trained in two stages: visual-language alignment, then domain specialisation with LoRA on 101K structured reports produced by an automated Gemma-3 labelling pipeline.

On a 5,062-sample held-out test set AgriPerceiver reaches a composite score of **0.810** with 99.7% schema compliance, against 0.590–0.615 for 7–8B general-purpose VLMs (LLaVA-NeXT-7B, InternVL2-8B, Qwen2-VL-7B). Pathology-type macro-F1 improves 1.9× over the strongest baseline (0.645 vs. 0.336) and severity MAE drops from 0.224 to 0.067, while only 9.2% of the model's parameters are trained.

---

## Publications

| Venue | Paper | Authors | Source |
|---|---|---|---|
| **ICA 2026** — International Conference on Agriculture-Centric Computation, IIT Ropar (Springer CCIS) | *AgriPerceiver: A Parameter-Efficient Vision-Language Model for Structured Macroscopic Crop Phenotyping* | Vatsal Khanna\*, Samridhi Saini\*, Davinder Singh\*, Kumkum Kwatra | [papers/ica26/](papers/ica26/) · [PDF](papers/ica26/main.pdf) · [talk](presentations/ica26/) |
| **ICML 2026** — 3rd Workshop on Multi-modal Foundation Models and LLMs for Life Sciences (FM4LS), Seoul | *AgriPerceiver: A Parameter-Efficient Vision-Language Model for Structured Macroscopic Crop Phenotyping* | Vatsal Khanna, Davinder Singh | [papers/icml26-fm4ls/](papers/icml26-fm4ls/) |

\* Equal contribution.

---

## Key Contributions

- **Perceiver Resampler bridge** — 28.5× visual token compression (3,645 → 128 latents) via cross-attention from learned queries, keeping LLM context length tractable without discarding high-resolution detail
- **AnyRes multi-scale tiling** — 5-tile decomposition (4 quadrants + global view) with learned per-tile position embeddings, so spatial provenance survives compression
- **Backbone-agnostic perception module** — the bridge is decoupled from the choice of vision encoder and language model
- **Two-stage curriculum, frozen backbones** — Stage 1 trains only the bridge (~391M parameters); Stage 2 adds rank-32 LoRA adapters (~35M), for ~426M trainable parameters in total (9.2% of 4.65B)
- **Automated agronomic labelling pipeline** — Gemma-3-12B-IT with JSON pre-fill forcing labels 117,635 raw images across 57 disease categories; confidence filtering keeps 101,301 high-quality samples
- **Comprehensive evaluation suite** — 9 metrics spanning structural validity, classification, regression, calibration, and semantic similarity, plus an LLM-as-judge consensus scorer

---

## Architecture

```
Input Image (H × W × 3)
        │
        ├─ AnyRes Tiling ─────────────────── 5 tiles × 384 × 384
        │
        ├─ SigLIP-SO400M-patch14-384 ─────── [5, 729, 1152]  patch features  (frozen)
        │
        ├─ TileEmbeddings ────────────────── + learned spatial position per tile
        │
        ├─ VisionProjector (MLP) ─────────── 1152 → 3072  (match LLM hidden dim)
        │
        ├─ PerceiverResampler ────────────── 3,645 tokens → 128 latents  (28.5× ↓)
        │   └─ 2 × [CrossAttention(Q=latents, KV=patches) + SelfAttention + GLU-FFN]
        │
        └─ Splice into Phi-3 ─────────────── [<image_start>  128 latents  <image_end>]
                                                            │
                                                            ▼
                                             Phi-3-mini-128k-instruct  (3.8B, frozen + LoRA r=32)
                                                            │
                                                            ▼
                                             Structured JSON diagnostic report
```

| Component | Backbone | Parameters | Training |
|---|---|---|---|
| Vision Encoder | SigLIP-SO400M-patch14-384 | ~400M | Frozen |
| Perception Bridge | TileEmbed + Projector + Perceiver (2 blocks) | ~391M | Stage 1 + 2 |
| Language Model | Phi-3-mini-128k-instruct | ~3.82B | Frozen; LoRA adapters (~35M) in Stage 2 |
| **Total trainable (Stage 2)** | — | **~426M** | **9.2% of 4.65B** |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for tensor shapes and design rationale.

---

## Results

All numbers below are from the ICA 2026 camera-ready paper ([papers/ica26/](papers/ica26/)).

### Main Comparison

Held-out test set of 5,062 samples. All baselines receive the same structured JSON prompt and are not fine-tuned.

| Model | Params | JSON% | Schema% | Type-F1 ↑ | Diag-FM ↑ | Sev-MAE ↓ | Sev-r ↑ | BERT-sym ↑ | **Composite ↑** |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-NeXT-7B | 7B | 99.9 | 99.9 | 0.134 | 0.001 | 0.344 | 0.226 | 0.847 | 0.590 |
| InternVL2-8B | 8B | 100.0 | 100.0 | 0.336 | 0.005 | 0.385 | 0.624 | 0.848 | 0.611 |
| Qwen2-VL-7B | 7B | 99.9 | 99.9 | 0.287 | 0.001 | 0.224 | 0.666 | 0.830 | 0.615 |
| **AgriPerceiver** | **4.6B** | 99.7 | 99.7 | **0.645** | **0.486** | **0.067** | **0.855** | **0.921** | **0.810** |

> The composite score $\mathcal{C}$ is a weighted sum: $0.20\,F_1^{\text{type}} + 0.15\,F_{\text{diag}} + 0.15\,B_{\text{sym}} + 0.10\,(1-\text{MAE}) + 0.10\,B_{\text{reas}} + 0.10\,B_{\text{act}} + 0.10\,J_{\text{val}} + 0.05\,(1-\text{ECE}) + 0.05\,S_{\text{cmp}}$. All components lie in [0, 1]. See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions.

### Per-Class Type-F1

| Pathology Class | Support | LLaVA-NeXT | InternVL2 | Qwen2-VL | AgriPerceiver |
|---|---|---|---|---|---|
| Fungal | 1,521 | 0.301 | 0.599 | 0.637 | **0.813** |
| Unknown | 1,535 | 0.493 | 0.689 | 0.634 | **0.827** |
| Bacterial | 692 | 0.000 | 0.003 | 0.025 | **0.560** |
| Deficiency | 618 | 0.006 | 0.475 | 0.386 | **0.605** |
| Pest | 351 | 0.000 | 0.227 | 0.017 | **0.541** |
| Viral | 345 | 0.006 | 0.022 | 0.022 | **0.525** |
| **Macro avg** | 5,062 | 0.134 | 0.336 | 0.287 | **0.645** |

### Ablation Study

Runtime ablations on the full test set (no retraining).

| Variant | Type-F1 ↑ | Diag-FM ↑ | Sev-MAE ↓ | Composite ↑ | Δ |
|---|---|---|---|---|---|
| Full model | 0.645 | 0.486 | 0.067 | 0.810 | — |
| − AnyRes (single tile) | 0.638 | 0.480 | 0.069 | 0.806 | −0.004 |
| − Tile embeddings | 0.647 | 0.483 | 0.067 | 0.810 | ≈ 0 |

The working results tracker, with per-metric breakdowns and analysis notes, is [docs/RESULTS.md](docs/RESULTS.md). Its composite scores (e.g. 0.717) predate the final BERTScore re-run; the table above is authoritative.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/VatsalKhanna5/agri-perceiver.git
cd agri-perceiver

# Install with core dependencies
pip install -e .

# Install with evaluation dependencies (BERTScore, scikit-learn, etc.)
pip install -e ".[eval]"

# Install with training dependencies (wandb, bitsandbytes)
pip install -e ".[train]"

# Full installation (all extras)
pip install -e ".[eval,train,dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, CUDA 11.8+ recommended.

---

## Model Weights & Data

> Model checkpoints and the training dataset will be released on HuggingFace.

| Artifact | Status | Link |
|---|---|---|
| `agri-perceiver/specialist-e3` — Stage-2 fine-tuned weights | Coming soon | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Coming%20Soon-FF9D00?style=flat-square&logo=huggingface&logoColor=white)](#) |
| `agri-perceiver/stage1-bridge` — Stage-1 connector weights | Coming soon | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Coming%20Soon-FF9D00?style=flat-square&logo=huggingface&logoColor=white)](#) |
| `agri-perceiver/canonical-dataset` — 101K training samples | Coming soon | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Coming%20Soon-FF9D00?style=flat-square&logo=huggingface&logoColor=white)](#) |

---

## Usage

### Single-Image Inference

```python
from agri_perceiver.inference.predictor import AgriPredictor

predictor = AgriPredictor("checkpoints/specialist_e3.pt")
report = predictor.predict("path/to/leaf.jpg")
print(report)
```

The returned `report` is a Pydantic `DiagnosticReport` (see [src/agri_perceiver/inference/schema.py](src/agri_perceiver/inference/schema.py)):

```json
{
  "diagnosis": "Alternaria leaf blight",
  "type": "fungal",
  "severity": 0.72,
  "confidence": 0.89,
  "symptoms": ["Circular necrotic lesions with concentric zonation", "Chlorotic halo around lesions"],
  "reasoning": "Concentric ring morphology and lesion distribution are consistent with...",
  "recommended_actions": ["Apply mancozeb (2 g/L) at 10-day intervals", "Remove and destroy infected debris"]
}
```

### CLI

```bash
# Single image inference
agri-predict --image leaf.jpg --checkpoint checkpoints/specialist_e3.pt

# Score a predictions file
agri-eval --predictions preds.jsonl --ground-truth gt.jsonl --output results/eval_results.json
```

### Training

```bash
# Stage 1 — Visual-language alignment (bridge only, ~391M trainable params)
python -m agri_perceiver.training.train_stage1 \
    --config configs/stage1_alignment.yaml \
    --data data/alignment/generic_alignment.jsonl \
    --data-root /path/to/images \
    --output checkpoints/stage1_connector_weights.pt

# Stage 2 — Agricultural specialisation (bridge + LoRA, ~426M trainable params)
python -m agri_perceiver.training.train_stage2 \
    --config configs/stage2_specialization.yaml \
    --data final_train_canonical.jsonl \
    --data-root canonical_dataset/processed_images/ \
    --bridge-weights checkpoints/stage1_connector_weights.pt \
    --output checkpoints/specialist_e{epoch}.pt
```

### Evaluation

```bash
# 1. Batch inference over the test split
python scripts/experiments/run_inference.py \
    --checkpoint checkpoints/specialist_e3.pt \
    --test-data data/test_split.jsonl \
    --image-root /path/to/processed_images \
    --output results/predictions.jsonl

# 2. Convert ground truth and score (9 metrics + composite)
python scripts/experiments/convert_gt.py data/test_split.jsonl data/test_split_gt.jsonl
agri-eval \
    --predictions results/predictions.jsonl \
    --ground-truth data/test_split_gt.jsonl \
    --output results/eval_results.json

# Baselines and ablations
python scripts/experiments/run_baseline_inference.py --model qwen2vl --test-data ... --output results/qwen2vl_predictions.jsonl
python scripts/experiments/run_ablation.py --ablation no_anyres --checkpoint checkpoints/specialist_e3.pt ...
```

The PBS scripts in [jobs/](jobs/) run these end to end on the cluster. [scripts/README.md](scripts/README.md) lists every script.

---

## Data Pipeline

The training corpus is built by a fully automated Gemma-3-12B-IT labelling pipeline:

```
117,635 raw images (57 disease categories)
        │
        ├─ generate_manifest.py  ──  Assigns canonical IDs (agri_000000..agri_117634)
        │
        ├─ gemma_pathologist.py  ──  Gemma-3-12B-IT (4-bit) with JSON pre-fill forcing
        │                            Batch size: 24 images  |  Output: structured JSON labels
        │
        └─ data_refining.py  ──────  Confidence filtering + schema normalisation
                                     Output: 101,301 samples with per-sample Gemma confidence weights
```

Pipeline scripts are in [data_pipeline/](data_pipeline/). See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for details.

---

## Repository Structure

```
agri-perceiver/
├── src/agri_perceiver/             # Core Python package
│   ├── model/                      #   AgriPerceiverVLM, PerceiverResampler, projector, tile embeddings, backbones
│   ├── data/                       #   Stage 1 / Stage 2 datasets and collation
│   ├── training/                   #   train_stage1.py, train_stage2.py
│   ├── inference/                  #   AgriPredictor API + DiagnosticReport schema  (agri-predict)
│   └── evaluation/                 #   9 metrics + composite, baselines, LLM judge  (agri-eval)
│
├── configs/                        # YAML configs: model, stage 1, stage 2, evaluation
├── data_pipeline/                  # Gemma-3 labelling: manifest → labels → refined dataset
│
├── scripts/
│   ├── experiments/                #   Batch inference, baselines, ablations, prediction checks
│   └── paper/                      #   Figure generation and camera-ready checks
├── jobs/                           # PBS job scripts for the HPC cluster
│
├── results/                        # Evaluation output (eval_results.json)
├── tests/                          # pytest suite
│
├── docs/                           # ARCHITECTURE, TRAINING, EVALUATION, DATA_PIPELINE, RESULTS
├── papers/
│   ├── ica26/                      #   ICA 2026 camera-ready (Springer LNCS/CCIS), with compiled PDF
│   └── icml26-fm4ls/               #   ICML 2026 FM4LS workshop paper
├── presentations/
│   └── ica26/                      #   ICA 2026 talk: HTML deck, presenter view, exported PDFs
│
├── assets/                         # README banner
└── pyproject.toml                  # Package config, dependencies, CLI entry points
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component-level architecture, tensor shapes end to end, Perceiver design rationale, multimodal fusion |
| [docs/TRAINING.md](docs/TRAINING.md) | Two-stage training protocol, LoRA configuration, hardware, optimiser settings |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metric definitions, baseline protocols, LLM-as-judge scoring axes and consensus |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Labelling pipeline: manifest generation, Gemma-3 inference, confidence filtering, normalisation |
| [docs/RESULTS.md](docs/RESULTS.md) | Working results tracker: per-metric breakdowns, ablation notes, sensitivity analysis |
| [papers/ica26/README.md](papers/ica26/README.md) | Building the camera-ready paper and regenerating its figures |
| [presentations/ica26/README.md](presentations/ica26/README.md) | Presenting and re-exporting the ICA 2026 talk |

---

## Workshop

This work was presented at the **3rd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences** (FM4LS), held on **July 11, 2026** at the **COEX Convention & Exhibition Center, Seoul, South Korea**, as part of **ICML 2026** (July 6–11, 2026).

The workshop brings together researchers at the intersection of multi-modal learning, foundation models, and life sciences to address challenges in modality fusion, cross-modal representation learning, scalable pretraining, and interpretability across biological domains, from genomics and proteomics to agricultural science.

| | |
|---|---|
| Workshop page | [icml2026fm4ls.github.io](https://icml2026fm4ls.github.io/index.html) |
| ICML 2026 | [icml.cc](https://icml.cc/) |
| Venue | COEX Convention & Exhibition Center, Room S317, Seoul, South Korea |
| Workshop date | July 11, 2026, 08:00–17:00 KST |

The work also appears at **ICA 2026**, the International Conference on Agriculture-Centric Computation ([ica26.annam.ai](https://ica26.annam.ai/)), IIT Ropar, 1–4 November 2026, published in Springer CCIS (paper 16).

---

## Acknowledgements

This work was carried out under the guidance of **Dr. Roshan Bodile** and **Dr. Rohit Singh** (Department of Electronics and Communication Engineering, Dr. B. R. Ambedkar National Institute of Technology Jalandhar), whose sustained mentorship shaped both the research direction and the technical rigor of this project.

Logistical support, domain expertise, and computational infrastructure were provided by **[Annam.ai](https://annam.ai)** — an AI Centre of Excellence in Agriculture established at the Indian Institute of Technology Ropar, under the leadership of **Dr. Pushpendra Pal Singh** (Project Director). The H200 GPU compute on which all training and evaluation runs were conducted was made available through Annam.ai. This work aims to contribute to the broader mission of Annam.ai in advancing AI-driven solutions for Indian and global agriculture.

---

## Citation

If you use AgriPerceiver in your research, please cite:

```bibtex
@inproceedings{khanna2026agriperceiver,
  title={AgriPerceiver: A Parameter-Efficient Vision-Language Model for Structured Macroscopic Crop Phenotyping},
  author={Khanna, Vatsal and Singh, Davinder},
  booktitle={ICML 2026 3rd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences}
}

@inproceedings{khanna2026agriperceiverica,
  title={AgriPerceiver: A Parameter-Efficient Vision-Language Model for Structured Macroscopic Crop Phenotyping},
  author={Khanna, Vatsal and Saini, Samridhi and Singh, Davinder and Kwatra, Kumkum},
  booktitle={International Conference on Agriculture-Centric Computation (ICA 2026)},
  series={Communications in Computer and Information Science},
  publisher={Springer},
  year={2026}
}
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).

The pre-trained backbone weights used in this work are subject to their respective licenses:
- SigLIP-SO400M: [Apache 2.0](https://huggingface.co/google/siglip-so400m-patch14-384)
- Phi-3-mini-128k-instruct: [MIT](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- Gemma-3-12B-IT (data pipeline only): [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

---

<div align="center">

*"The difficulty lies not so much in developing new ideas as in escaping from old ones."*
— John Maynard Keynes

</div>
