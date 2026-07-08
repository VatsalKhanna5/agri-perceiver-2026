# AgriPerceiver — ICA 2026 submission (Springer LNCS/CCIS, Overleaf-ready)

This folder is a self-contained, **Overleaf-ready** Springer LNCS project for
the ICA 2026 conference (<https://ica26.annam.ai/>). It is prepared for the
**double-blind** review track: all author-identifying information has been
removed.

## Contents
| File | Purpose |
|------|---------|
| `main.tex` | The paper (Springer LNCS class, single column). |
| `references.bib` | Bibliography (BibTeX). |
| `llncs.cls` | Springer LNCS document class (v2.24) — CCIS uses the same class. |
| `splncs04.bst` | Springer LNCS BibTeX style. |
| `figures/architecture.pdf` | System architecture diagram (vector PDF). |
| `figures/perclass_f1.pdf` | Per-class pathology-type F1, all four models. |
| `figures/confusion.pdf` | Row-normalised confusion matrix (ours). |
| `figures/calibration.pdf` | Reliability diagram (ours). |
| `figures/flops.pdf` | Attention-FLOP scaling: full self-attn vs Perceiver. |
| `figures/training_curves.pdf` | Stage-1 / Stage-2 training loss curves. |

All plot figures are regenerated from the real evaluation output
(`results/eval_results.json`) by `scripts/make_figs.py` in the repo root.

## Building on Overleaf
1. Create a new project → **Upload Project** → zip and upload this whole folder.
2. Set the main document to `main.tex`, compiler **pdfLaTeX**.
3. It compiles as-is (no extra packages needed).

## Building locally
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Compliance notes (per the Springer author instructions)
- **Format:** Springer LNCS class (`llncs`), the basis for the CCIS series.
- **Length:** targets the 12–15-page full-paper range once compiled.
- **Double-blind:** authors, affiliation, emails, and identifying
  acknowledgements are withheld (`\author{Anonymous Author(s)}`).
- **Headings:** title-case, only two levels numbered.
- **Captions:** table captions above, figure captions below.
- **Figures:** vector PDFs (no rasterised line art).
- **Appendix:** placed **before** the references, as Springer requires.
- **References:** numeric `\cite`, formatted with `splncs04.bst`.
- **Disclosure of Interests:** included via the `credits` environment.

## Before camera-ready (after acceptance)
Restore, in `main.tex`:
- `\author{...}`, `\authorrunning{...}`, `\institute{...}` with real names,
  affiliations, emails, and ORCIDs.
- The acknowledgements text under `\subsubsection{\ackname}` (funding,
  computing infrastructure, mentorship) that was withheld for anonymity.
