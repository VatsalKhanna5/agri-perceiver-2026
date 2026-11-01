# Getting Started

## Installation

```bash
git clone https://github.com/VatsalKhanna5/agri-perceiver.git
cd agri-perceiver

pip install -e .                    # core: model, training, inference
pip install -e ".[eval]"            # + BERTScore, scikit-learn, plotting
pip install -e ".[train]"           # + wandb, bitsandbytes
pip install -e ".[eval,train,dev]"  # everything, incl. pytest and ruff
```

**Requirements:** Python ≥ 3.10 and PyTorch ≥ 2.1. A CUDA GPU is recommended. Inference needs
about 16 GB of VRAM, since the bfloat16 weights alone take about 8 GB.

Install the `eval` extra before scoring predictions. Without `bert-score`, the BERTScore terms
come out as 0 and the composite score is not comparable to the paper.

## Model weights and data

The checkpoints and the training dataset will be released on HuggingFace.

| Artifact | Contents | Status |
|---|---|---|
| `agri-perceiver/specialist-e3` | Stage-2 model (bridge + LoRA, full state dict) | Coming soon |
| `agri-perceiver/stage1-bridge` | Stage-1 bridge weights (tile embeddings, projector, Perceiver) | Coming soon |
| `agri-perceiver/canonical-dataset` | 101,301 image–report training samples | Coming soon |

The backbones, `google/siglip-so400m-patch14-384` and `microsoft/Phi-3-mini-128k-instruct`,
download from HuggingFace on first use.

## Inference

### Python

```python
from agri_perceiver.inference.predictor import AgriPredictor

predictor = AgriPredictor("checkpoints/specialist_e3.pt")
report = predictor.predict("path/to/leaf.jpg")
print(report.model_dump_json(indent=2))
```

`predict()` returns a `DiagnosticReport`. If the output can't be parsed, it returns
`{"raw_output": ..., "parse_error": True}` instead. Pass `return_raw=True` to get the generated
text. `batch_predict()` takes a list of paths. Pass `use_lora=False` to the constructor to load
Stage-1 bridge weights.

### Command line

```bash
agri-predict --image leaf.jpg --checkpoint checkpoints/specialist_e3.pt
agri-predict --image leaf.jpg --checkpoint checkpoints/stage1_connector_weights.pt --no-lora --raw
```

### Output schema

Reports follow `DiagnosticReport` in
[src/agri_perceiver/inference/schema.py](../src/agri_perceiver/inference/schema.py):

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

| Field | Type | Notes |
|---|---|---|
| `diagnosis` | string | Primary pathogen or condition |
| `type` | string | One of `fungal`, `bacterial`, `viral`, `pest`, `deficiency`, `unknown` |
| `severity` | float ∈ [0, 1] | 0 = none, 1 = critical |
| `confidence` | float ∈ [0, 1] | Model certainty in the diagnosis |
| `symptoms` | list[string] | Observed visual symptoms |
| `reasoning` | string | Pathological reasoning |
| `recommended_actions` | list[string] | Suggested treatments |

## Next steps

- [EVALUATION.md](EVALUATION.md#running-evaluation): batch inference over a test split and scoring
- [TRAINING.md](TRAINING.md): training both stages from scratch
- [ARCHITECTURE.md](ARCHITECTURE.md): how the model works
