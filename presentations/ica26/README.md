# AgriPerceiver: ICA 2026 talk

The talk for paper 16 at ICA 2026 (IIT Ropar, 1–4 Nov 2026). The slot is 12 minutes of talk and 3 minutes of Q&A.
The deck is 15 slides plus one backup slide, and every number in it comes from the camera-ready paper.

**Submission file:** `export/AgriPerceiver-ICA26.pdf`. It has 16 pages at 16:9, with every build shown in its final state. The organisers asked for it by **Thursday 29 October 2026**.

## Present

```bash
cd presentations/ica26
python -m http.server 8765
```

Open <http://localhost:8765>, press **F** for fullscreen, then press **S** to open the presenter view in a second window.
The deck needs to be served from a local server like this. Opening `index.html` straight from disk still shows the slides, but the presenter sync and the PDF button need http.
Fonts and figures are bundled in the folder, so the deck works offline at the venue.

| Key | Deck | Presenter view |
| --- | --- | --- |
| → Space PgDn Enter | next build / slide | drives the audience screen |
| ← PgUp Backspace | previous | same |
| F | fullscreen | |
| S | open presenter view | |
| O | overview grid (click a slide to jump) | |
| B or . | black screen (clicker "blank" key) | |
| D | download PDF | |
| T / R | | start-pause / reset the clock |
| Q | | switch between the script and Q&A prep |

Slide 6 (architecture) builds in three clicks. Every other slide builds itself when it opens.

## Presenter view

`presenter.html` shows the current slide and its build state, the next slide, the speaker script with an anchor line for each slide and click cues, a 12:00 clock that tells you if you are ahead of or behind each slide's target, and Q&A prep.
The per-slide targets add up to 11:05, which leaves about a minute of buffer.

`presenter.html?script` is a printable version of the script. It is also exported as `export/AgriPerceiver-ICA26-speaker-script.pdf`.

## Files

```
index.html               the slides
presenter.html           presenter view and printable script
assets/js/data.js        every number, with its source in the paper
assets/js/charts.js      the diagrams and charts, drawn as SVG from data.js
assets/js/notes.js       speaker notes, timings, Q&A
assets/js/deck.js        scaling, navigation, builds, sync, overview, PDF
assets/css/              design tokens (from the Field), slide layouts, local fonts
assets/figures/          wireframe plates from duckn.rocks-field
tools/export.mjs         rebuilds both PDFs with headless Chrome
```

## Re-export the PDFs after editing

```bash
cd presentations/ica26/tools
npm install
npm run export
```

`export.mjs` serves the folder on its own and uses the installed Chrome or Edge. Set `CHROME=` to use a different browser.
You can also open the deck in a browser and use **Print → Save as PDF**, with margins set to none and background graphics on.

## Data notes

- The numbers come from `papers/ica26/main.tex` (the camera-ready source) and from `results/eval_results.json`.
  `docs/RESULTS.md` has older composite scores (0.717 and 0.527) and is not used.
- The per-class F1 scores for the baselines come from `scripts/paper/make_figs.py`, the source of Fig. 2.
  For pest, InternVL2 scores **0.227**. The paper's text says baselines score pest F1 ≤ 0.017, which only holds for Qwen2-VL and LLaVA.
  The slides only claim "≤ 0.025" for bacterial and viral, and the Q&A notes include an answer for this.
- The chart palette passed the dataviz validator against the paper surface `#FDFCF6`.
  The validator checks lightness, chroma, separation for colour-blind and normal vision, and contrast.
  The baselines also use hatch patterns, so the charts still work in greyscale.
