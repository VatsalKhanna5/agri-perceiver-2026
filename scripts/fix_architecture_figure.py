"""Correct factual/spelling errors in the draw.io-exported architecture figure.

Fixes, all verified against src/agri_perceiver/model/perceiver_resampler.py
(PerceiverBlock: CrossAttn -> SelfAttn -> FFN; heads=24, dim=3072 -> head dim 128)
and src/agri_perceiver/model/agri_vlm.py (splice_and_forward):

  1. "Perciver Block 1/2"  -> "Perceiver Block 1/2"      (spelling, x2)
  2. "Slice & Forward"     -> "Splice & Forward"         (matches splice_and_forward)
  3. "Cross Attention"     -> "Cross + Self Attention"   (self-attention stage was missing)
  4. "Dim = 24"            -> "Dim = 128"                (head dim = 3072/24 = 128)

The stale strings are REMOVED from the text layer via redaction (not merely
painted over), so copy-paste, search and accessibility tools see the corrected
wording. Items 1-3 are real text; item 4 is vector line art in the draw.io
export, so it is covered with the box's own fill colour and redrawn.
"""
import os
import shutil
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'paper-ica26', 'figures')
SRC = os.path.join(FIGS, 'architecture_orig.pdf')   # pristine draw.io export
DST = os.path.join(FIGS, 'architecture.pdf')

GREY_BG = (0.9608, 0.9608, 0.9608)
BLUE_BG = (0.8549, 0.9098, 0.9882)
GREY_TXT = (0.612, 0.612, 0.612)
BLACK = (0, 0, 0)
# Publishers require every font embedded, and PyMuPDF's base-14 aliases are not.
# TeX Gyre Termes/Cursor are freely licensed (GUST FL) and metrically compatible
# with Times/Courier, so the width arithmetic below is unaffected.
TG = os.environ.get('TEXGYRE_DIR', os.path.expandvars(
    r'%LOCALAPPDATA%\Programs\MiKTeX\fonts\opentype\public\tex-gyre'))
FF = {'tgbold': os.path.join(TG, 'texgyretermes-bold.otf'),
      'tgital': os.path.join(TG, 'texgyretermes-italic.otf'),
      'tgmono': os.path.join(TG, 'texgyrecursor-regular.otf')}
for _n, _p in FF.items():
    if not os.path.exists(_p):
        raise SystemExit(f'TeX Gyre font not found: {_p}\n'
                         'Install the tex-gyre package or set TEXGYRE_DIR.')
METRIC = {'tgbold': 'tibo', 'tgital': 'tiit', 'tgmono': 'cour'}
BOLD, ITAL, MONO = 'tgbold', 'tgital', 'tgmono'

shutil.copyfile(SRC, DST)
doc = pymupdf.open(DST)
page = doc[0]


def w(text, size, font=BOLD):
    return pymupdf.get_text_length(text, fontname=METRIC[font], fontsize=size)


def put(point, text, font, size, color, **kw):
    page.insert_text(point, text, fontname=font, fontfile=FF[font],
                     fontsize=size, color=color, **kw)


# ---- 1. strip the stale strings from the text layer ------------------------
for b in page.get_text('dict')['blocks']:
    for line in b.get('lines', []):
        for s in line['spans']:
            if s['text'].strip() in ('Perciver', 'Slice & Forward', 'Cross'):
                page.add_redact_annot(pymupdf.Rect(s['bbox']) + (-1, -1, 1, 1))
page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                      graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                      text=pymupdf.PDF_REDACT_TEXT_REMOVE)

# ---- 2. "Perceiver Block 1 / 2" -------------------------------------------
for x_blk0, x_blk1, base in ((942.1, 982.4, 280.62), (1027.7, 1068.0, 281.32)):
    cx = (x_blk0 + x_blk1) / 2
    page.draw_rect(pymupdf.Rect(cx - 32, base - 12.0, cx + 32, base + 3.5),
                   color=None, fill=GREY_BG, width=0)
    put((cx - w('Perceiver', 12.63) / 2, base), 'Perceiver',
        BOLD, 12.63, GREY_TXT)

# ---- 3. "Splice & Forward" (rotated 90 deg CCW) ---------------------------
new, size = 'Splice & Forward', 12.63
size = min(size, size * 89.7 / w(new, size))          # keep original extent
page.draw_rect(pymupdf.Rect(1104.0, 150.0, 1122.0, 252.0),
               color=None, fill=GREY_BG, width=0)
ycentre = (156.9 + 246.6) / 2
put((1117.34, ycentre + w(new, size) / 2), new, BOLD, size, BLACK, rotate=90)

# ---- 4. "Cross + Self" and "Dim = 128" in both Perceiver blocks -----------
for dx in (0.0, 85.93):                               # blue box 1 -> box 2 offset
    x0, x1 = 938.1 + dx, 987.6 + dx
    cx = (x0 + x1) / 2

    s, size = 'Cross + Self', 8.21
    size = min(size, size * (x1 - x0 - 5) / w(s, size))
    page.draw_rect(pymupdf.Rect(x0 + 1.0, 126.0, x1 - 1.0, 138.6),
                   color=None, fill=BLUE_BG, width=0)
    put((cx - w(s, size) / 2, 137.9), s, BOLD, size, BLACK)

    s2, size2 = 'Dim = 128', 7.4                      # vector line art: cover + redraw
    page.draw_rect(pymupdf.Rect(x0 + 6, 165.8, x1 - 6, 174.6),
                   color=None, fill=BLUE_BG, width=0)
    put((cx - w(s2, size2, ITAL) / 2, 172.9), s2, ITAL, size2, BLACK)

# ---- 5. legible output sample ---------------------------------------------
# The export filled the "Test-Results" box with a full JSON dump at 3.79 pt.
# Scaled into the paper that is ~1.6 pt -- grey noise in print. Replace it with
# a short, legible skeleton of the same seven-field schema.
JSON_BOX = pymupdf.Rect(1325.3, 115.4, 1488.7, 290.8)
page.add_redact_annot(JSON_BOX + (2, 2, -2, -2))
page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                      graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                      text=pymupdf.PDF_REDACT_TEXT_REMOVE)
page.draw_rect(JSON_BOX + (1.5, 1.5, -1.5, -1.5), color=None, fill=(1, 1, 1), width=0)

SNIPPET = [
    '{',
    ' "diagnosis": "...",',
    ' "type": "fungal",',
    ' "severity": 0.40,',
    ' "confidence": 0.85,',
    ' "symptoms": [...],',
    ' "reasoning": "...",',
    ' "recommended_actions":',
    '     [...]',
    '}',
]
PAD = 7.0
avail_w = JSON_BOX.width - 2 * PAD
avail_h = JSON_BOX.height - 2 * PAD
widest = max(pymupdf.get_text_length(l, fontname='cour', fontsize=10) for l in SNIPPET)
JSIZE = min(10 * avail_w / widest, avail_h / (len(SNIPPET) * 1.28))
LEAD = JSIZE * 1.28
y = JSON_BOX.y0 + PAD + JSIZE + (avail_h - len(SNIPPET) * LEAD) / 2
for ln in SNIPPET:
    put((JSON_BOX.x0 + PAD, y), ln, MONO, JSIZE, BLACK)
    y += LEAD
print(f'  JSON snippet at {JSIZE:.1f} pt native (~{JSIZE * 0.42:.1f} pt on page)')

doc.save(DST, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
print('patched ->', DST)
