"""Split the wide architecture figure into two stacked panels.

The source diagram is 1509x459 pt (3.3:1). Included at LNCS \textwidth (346 pt)
its lettering renders at 0.9-4.3 pt, far below Springer's ~8 pt / 2 mm minimum
for print. Cutting it into two panels roughly halves the width each panel must
be scaled down by, bringing the labels back into a legible range.

The cut falls in the gap between the Tile-Embeddings stage and the Vision
Projector, with a small overlap so the connecting arrow stays visible in both.
Each panel is then cropped to its own content, removing dead vertical space.
"""
import os
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, 'papers', 'ica26', 'figures')
SRC = os.path.join(FIGS, 'architecture.pdf')
OUT_A = os.path.join(FIGS, 'architecture_a.pdf')
OUT_B = os.path.join(FIGS, 'architecture_b.pdf')

CUT = 757.0
OVERLAP = 6.0
PAD = 5.0

src = pymupdf.open(SRC)
page = src[0]


def content_bbox(x_lo, x_hi):
    xs, ys = [], []
    items = []
    for dr in page.get_drawings():
        items.append(dr['rect'])
    for b in page.get_text('dict')['blocks']:
        items.append(pymupdf.Rect(b['bbox']))
    for img in page.get_images():
        items.extend(page.get_image_rects(img[0]))
    full_w = page.rect.width
    for r in items:
        if r.width >= full_w * 0.99:        # skip the page-sized background
            continue
        if r.x1 <= x_lo or r.x0 >= x_hi:    # outside this half
            continue
        xs += [max(r.x0, x_lo), min(r.x1, x_hi)]
        ys += [r.y0, r.y1]
    return min(xs), min(ys), max(xs), max(ys)


for out, lo, hi in ((OUT_A, 0.0, CUT + OVERLAP), (OUT_B, CUT - OVERLAP, page.rect.width)):
    cx0, cy0, cx1, cy1 = content_bbox(lo, hi)
    clip = pymupdf.Rect(max(lo, cx0 - PAD), max(0, cy0 - PAD),
                        min(hi, cx1 + PAD), min(page.rect.height, cy1 + PAD))
    doc = pymupdf.open()
    new = doc.new_page(width=clip.width, height=clip.height)
    new.show_pdf_page(new.rect, src, 0, clip=clip)
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    print(f'{out.rsplit(chr(92), 1)[-1]:20s} clip={[round(v, 1) for v in clip]} '
          f'size={clip.width:.0f}x{clip.height:.0f} aspect={clip.width / clip.height:.2f}')
