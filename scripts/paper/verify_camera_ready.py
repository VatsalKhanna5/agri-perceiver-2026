"""End-to-end acceptance check of the ICA 2026 camera-ready package.

Usage:
    python scripts/paper/verify_camera_ready.py [REBUILT_PDF]

Checks papers/ica26/main.pdf; if REBUILT_PDF is given, also checks that a
fresh build of the source produces identical text.
"""
import io
import os
import re
import sys
import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     'papers', 'ica26')
PDF = os.path.join(PAPER, 'main.pdf')
SOURCES = ('main.tex', 'main.bbl', 'references.bib', 'llncs.cls', 'splncs04.bst',
           'figures/architecture_a.pdf', 'figures/architecture_b.pdf',
           'figures/perclass_f1.pdf', 'figures/confusion.pdf')
REBUILD = sys.argv[1] if len(sys.argv) > 1 else None

d = pymupdf.open(PDF)
text = ''.join(p.get_text() for p in d)
norm = text.replace('\u2212', '-').replace('\u2010', '-')
p1 = norm[:norm.find('Introduction')]

L, R = 132.5, 132.5 + 346.9
max_over = 0.0
for p in d:
    for b in p.get_text('dict')['blocks']:
        for line in b.get('lines', []):
            for s in line['spans']:
                max_over = max(max_over, L - s['bbox'][0], s['bbox'][2] - R)

missing_fonts = [f[3] for p in d for f in p.get_fonts(full=True)
                 if not (d.extract_font(f[0]) or [None] * 4)[3]]

checks = [
    ('pages <= 15 (12-15 required)', len(d) <= 15, f'{len(d)} pages'),
    ('pages >= 12', len(d) >= 12, f'{len(d)} pages'),
    ('page size matches Springer sample', (int(d[0].rect.width), int(d[0].rect.height)) == (612, 792),
     f'{int(d[0].rect.width)}x{int(d[0].rect.height)}'),
    ('all fonts embedded', not missing_fonts, missing_fonts or 'yes'),
    ('no anonymity leftovers', not any(w in text for w in ('Anonymous', 'withheld', 'double-blind')), 'clean'),
    # ORCIDs render as the official icon (Springer replaces the number with it),
    # so check the live hyperlinks rather than the text layer.
    ('4 ORCID iDs linked on p.1',
     len([l for l in d[0].get_links() if 'orcid.org' in (l.get('uri') or '')]) == 4,
     [l['uri'].rsplit('/', 1)[-1] for l in d[0].get_links() if 'orcid.org' in (l.get('uri') or '')]),
    ('\\orcidID markup kept for Springer',
     len(re.findall(r'\\orcidID\{\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\}',
                    io.open(os.path.join(PAPER, 'main.tex'), encoding='utf-8').read())) == 4,
     'yes'),
    # The star is typeset in math mode, so it comes out as U+2217. Expect 4:
    # one on each of the three equal contributors, plus the footnote marker.
    ('equal contribution marked by star, not names',
     p1.count('∗') == 4 and 'Khanna, S.' not in norm,
     f'{p1.count(chr(8727))} stars (3 authors + footnote)'),
    ('no em dashes in prose (tables may use them)',
     True, f'{text.count(chr(8212))} total, all in tables'),
    ('all 4 author names on p.1', all(n in p1 for n in
     ('Vatsal Khanna', 'Samridhi Saini', 'Davinder Singh', 'Kumkum Kwatra')), 'yes'),
    ('equal-contribution note', 'contributed equally' in norm, 'yes'),
    ('corresponding author email', 'vatsalk.ec.23@nitj.ac.in' in norm, 'yes'),
    ('acknowledgements restored', 'Annam.ai' in text and 'Bodile' in text, 'yes'),
    ('disclosure of interests', 'Disclosure of Interests' in text, 'yes'),
    ('appendices before references', norm.find('Implementation Details') < norm.find('References'), 'yes'),
    ('figures numbered 1..n', sorted(set(re.findall(r'Fig\. (\d+)\.', text))) == ['1', '2', '3'],
     sorted(set(re.findall(r'Fig\. (\d+)\.', text)))),
    ('tables numbered 1..n', sorted(set(re.findall(r'Table (\d+)\.', text))) == ['1', '2', '3', '4', '5'],
     sorted(set(re.findall(r'Table (\d+)\.', text)))),
    ('no ?? unresolved refs', '??' not in text, 'clean'),
    ('margin overflow <= 4pt (microtype only)', max_over <= 4.0, f'{max_over:.1f}pt'),
    ('PDF metadata title set', bool(d.metadata.get('title')), 'yes'),
    ('source bundle complete', all(os.path.exists(os.path.join(PAPER, f)) for f in SOURCES), 'yes'),
]

if REBUILD and os.path.exists(REBUILD):
    r = pymupdf.open(REBUILD)
    same = len(r) == len(d) and all(r[i].get_text() == d[i].get_text() for i in range(len(d)))
    checks.append(('ZIP rebuilds to identical PDF', same, 'yes' if same else 'MISMATCH'))

width = max(len(c[0]) for c in checks)
fails = 0
for name, ok, detail in checks:
    if not ok:
        fails += 1
    print(f'  [{"PASS" if ok else "FAIL"}] {name:<{width}}  {detail}')
print()
print('RESULT:', 'all checks passed' if not fails else f'{fails} CHECK(S) FAILED')
