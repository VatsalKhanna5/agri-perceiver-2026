# AgriPerceiver — ICA 2026 camera-ready (Springer LNCS/CCIS)

Camera-ready source for paper **id 16**, *"AgriPerceiver: A Parameter-Efficient
Vision-Language Model for Structured Macroscopic Crop Phenotyping"*, accepted to
ICA 2026 (<https://ica26.annam.ai/>). Author identification has been restored
for the final version.

## Contents
| File | Purpose |
|------|---------|
| `main.tex` | The paper (Springer LNCS class, single column). |
| `references.bib` | Bibliography (BibTeX). |
| `llncs.cls` | Springer LNCS document class (v2.24) — CCIS uses the same class. |
| `splncs04.bst` | Springer LNCS BibTeX style. |
| `figures/architecture_a.pdf` | Fig. 1(a): tiling and encoding front end. |
| `figures/architecture_b.pdf` | Fig. 1(b): compression and generation. |
| `figures/perclass_f1.pdf` | Fig. 2: per-class pathology-type F1, all four models. |
| `figures/confusion.pdf` | Fig. 3: row-normalised confusion matrix (ours). |
| `figures/architecture.pdf` | Unsplit master of Fig. 1 (not used by `main.tex`). |

Plot figures are regenerated from the real evaluation output
(`results/eval_results.json`) by `scripts/make_figs.py` in the repo root.

## Building
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
Or `latexmk -pdf main`. On Overleaf: upload the folder as a project, set
`main.tex` as the main document, compiler **pdfLaTeX**. No extra packages are
needed — `llncs.cls` and `splncs04.bst` are bundled.

If a local build fails with *"auto expansion is only possible with scalable
fonts"*, install the `cm-super` Type 1 fonts (`mpm --install cm-super` on
MiKTeX); this does not affect Overleaf.

## Camera-ready compliance checklist
- **Format:** Springer LNCS class (`llncs`), the basis for the CCIS series.
- **Length:** 15 pages including references — within the 12–15 full-paper range.
- **Authors:** full names, both affiliations, corresponding-author email, and
  ORCID iDs for all four authors; equal contribution recorded in a first-page
  footnote.
- **Headings:** title-case, only two levels numbered.
- **Captions:** table captions above, figure captions below.
- **Figures:** vector PDFs, no rasterised line art.
- **Greyscale-safe:** no figure relies on colour alone. Bar series carry
  distinct hatch patterns and are ordered by luminance; the confusion matrix
  uses a luminance-monotonic map and prints its value in every cell.
- **Appendices:** placed **before** the references, as Springer requires.
- **References:** numeric `\cite`, formatted with `splncs04.bst`; proper nouns
  are brace-protected in `references.bib` so the style does not lowercase them.
- **Acknowledgements and Disclosure of Interests:** included via the `credits`
  environment.
