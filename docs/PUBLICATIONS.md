# Publications

The two papers share a title. Both describe *AgriPerceiver: A Parameter-Efficient
Vision-Language Model for Structured Macroscopic Crop Phenotyping*.

## ICA 2026 — Springer CCIS

**International Conference on Agriculture-Centric Computation (ICA 2026)**, IIT Ropar, India,
1–4 November 2026. Proceedings in Springer *Communications in Computer and Information Science*.
Paper id 16.

**Authors:** Vatsal Khanna\*, Samridhi Saini\*, Davinder Singh\*, Kumkum Kwatra
(\* equal contribution)

| | |
|---|---|
| Conference | [ica26.annam.ai](https://ica26.annam.ai/) |
| Camera-ready PDF | [papers/ica26/main.pdf](../papers/ica26/main.pdf) |
| LaTeX source | [papers/ica26/](../papers/ica26/) (build and figure instructions in its [README](../papers/ica26/README.md)) |
| Talk | [presentations/ica26/](../presentations/ica26/) — HTML deck, presenter view, [slides PDF](../presentations/ica26/export/AgriPerceiver-ICA26.pdf) |

## ICML 2026 — FM4LS Workshop

**3rd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences
(FM4LS)**, held at ICML 2026 (6–11 July 2026).

**Authors:** Vatsal Khanna, Davinder Singh

| | |
|---|---|
| Paper | [icml.cc/virtual/2026/72095](https://icml.cc/virtual/2026/72095) |
| Workshop page | [icml2026fm4ls.github.io](https://icml2026fm4ls.github.io/index.html) |
| ICML 2026 | [icml.cc](https://icml.cc/) |
| Venue | COEX Convention & Exhibition Center, Room S317, Seoul, South Korea |
| Date | 11 July 2026, 08:00–17:00 KST |
| LaTeX source | [papers/icml26-fm4ls/](../papers/icml26-fm4ls/) (needs the ICML 2026 style files) |

The workshop brings together researchers working across multi-modal learning, foundation models
and the life sciences. Its topics include modality fusion, cross-modal representation learning,
scalable pretraining, and interpretability in fields from genomics and proteomics to agricultural
science.

## Citation

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

## Reproducing paper figures

The scripts in [scripts/paper/](../scripts/paper/) regenerate the figures from
[results/eval_results.json](../results/eval_results.json) and check the camera-ready package:

```bash
python scripts/paper/make_figs.py              # per-class F1, confusion, calibration, FLOPs
python scripts/paper/verify_camera_ready.py    # 21 acceptance checks on papers/ica26/main.pdf
```
