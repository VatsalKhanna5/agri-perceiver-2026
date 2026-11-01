# Documentation

| Document | What it covers |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, model weights, single-image inference (Python and CLI), output schema |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, tensor shapes end to end, splice-and-forward fusion, design rationale |
| [TRAINING.md](TRAINING.md) | Two-stage curriculum, commands, LoRA and loss details, hardware |
| [EVALUATION.md](EVALUATION.md) | Metric suite, baselines, LLM-as-judge, running inference and scoring |
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | Gemma-3 labelling pipeline: manifest, labelling, confidence filtering |
| [RESULTS.md](RESULTS.md) | Main results, per-class breakdown, ablations, composite robustness, analysis |
| [PUBLICATIONS.md](PUBLICATIONS.md) | Papers, venues, talk, citations, reproducing paper figures |

Other READMEs: [scripts/](../scripts/README.md) (every script),
[papers/ica26/](../papers/ica26/README.md) (building the camera-ready paper) and
[presentations/ica26/](../presentations/ica26/README.md) (giving and exporting the talk).

## Repository map

```
agri-perceiver/
├── src/agri_perceiver/             # Python package
│   ├── model/                      #   AgriPerceiverVLM, PerceiverResampler, VisionProjector,
│   │                               #   TileEmbeddings, backbone loaders
│   ├── data/                       #   Stage 1 / Stage 2 datasets and collation
│   ├── training/                   #   train_stage1.py, train_stage2.py
│   ├── inference/                  #   AgriPredictor + DiagnosticReport schema   (agri-predict)
│   └── evaluation/                 #   metrics, composite, baselines, LLM judge  (agri-eval)
│
├── configs/                        # Reference hyperparameters (YAML)
├── data_pipeline/                  # generate_manifest → gemma_pathologist → data_refining
│
├── scripts/
│   ├── experiments/                #   batch inference, GT conversion, baselines, ablations
│   └── paper/                      #   figure generation, camera-ready checks
├── jobs/                           # PBS job scripts for the HPC cluster
│
├── results/                        # eval_results.json
├── tests/                          # pytest suite (model components, metrics)
│
├── docs/                           # this folder
├── papers/
│   ├── ica26/                      #   ICA 2026 camera-ready source + PDF (Springer LNCS/CCIS)
│   └── icml26-fm4ls/               #   ICML 2026 FM4LS workshop source
├── presentations/ica26/            # ICA 2026 talk: HTML deck, presenter view, PDFs
│
├── assets/                         # README banner
└── pyproject.toml                  # package metadata, dependencies, CLI entry points, tool config
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest                                  # runs from a fresh checkout; src/ is on the path
ruff check src tests scripts data_pipeline
```

CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs both on every push and PR to
`master`.
