# scripts/

Standalone scripts, run from the repository root. The reusable code lives in
`src/agri_perceiver/`; these scripts drive it for experiments and the papers.

## experiments/

Inference and evaluation runs, normally launched by the PBS jobs in `jobs/`.

| Script | Purpose |
|---|---|
| `run_inference.py` | Batch inference with AgriPerceiver over the test split → predictions JSONL (`--resume`, `--max-samples`) |
| `convert_gt.py` | Convert `test_split.jsonl` into the ground-truth format `agri-eval` expects |
| `validate_predictions.py` | Quick sanity report on a predictions file: JSON validity, schema coverage, distributions, timing |
| `run_baseline_inference.py` | Batch inference for baseline VLMs (`qwen2vl`, `llava_next`, `internvl2`) |
| `download_baselines.py` | Pre-fetch baseline weights into the HuggingFace cache |
| `patch_internlm2.py` | Patch cached InternVL2-8B remote code for transformers ≥ 4.50 |
| `run_ablation.py` | Runtime ablations without retraining (`no_lora`, `no_anyres`, `no_tile_embed`) |

## paper/

Figures and checks for the papers in `papers/`.

| Script | Purpose |
|---|---|
| `make_figs.py` | Plot figures for `papers/ica26/figures/` from `results/eval_results.json` |
| `fix_architecture_figure.py` | Correct labels in the draw.io architecture export (needs PyMuPDF + TeX Gyre fonts) |
| `split_architecture_figure.py` | Split the architecture figure into two stacked panels for LNCS width |
| `verify_camera_ready.py` | Acceptance checks on the ICA 2026 camera-ready PDF and source bundle |
| `plot_training_curves.py` | Training-loss curves for `papers/icml26-fm4ls/figures/` from the training logs |
