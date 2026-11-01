/* Diagrams and charts, drawn as SVG from window.AP.
   Conventions: marks carry `data-tip` for the hover layer; animated marks use
   the .grow-y / .grow-x / .draw / .pop classes from deck.css, with --i for
   stagger. Text inside charts wears ink, never the series colour. */

(function () {
  const AP = window.AP;
  const fmt = (v, d = 3) => v.toFixed(d);
  const thousands = (n) => n.toLocaleString("en-US");
  const SERIES = {
    llava: { name: "LLaVA-NeXT-7B", color: "var(--s-llava)", hatch: "h45" },
    intern: { name: "InternVL2-8B", color: "var(--s-intern)", hatch: "h135" },
    qwen: { name: "Qwen2-VL-7B", color: "var(--s-qwen)", hatch: "hx" },
    ours: { name: "AgriPerceiver", color: "var(--leaf)", hatch: null },
  };

  // Hatch textures: identity never rides on colour alone (greyscale, CVD, print).
  const hatchDefs = (p) => `
    <defs>
      <pattern id="${p}-h45" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="9" height="9" fill="var(--s-llava)"/><line x1="0" y1="0" x2="0" y2="9" stroke="rgba(253,252,246,.55)" stroke-width="3"/></pattern>
      <pattern id="${p}-h135" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
        <rect width="9" height="9" fill="var(--s-intern)"/><line x1="0" y1="0" x2="0" y2="9" stroke="rgba(253,252,246,.45)" stroke-width="2.5"/></pattern>
      <pattern id="${p}-hx" width="10" height="10" patternUnits="userSpaceOnUse">
        <rect width="10" height="10" fill="var(--s-qwen)"/><path d="M0 0L10 10M10 0L0 10" stroke="rgba(253,252,246,.45)" stroke-width="1.6"/></pattern>
      <pattern id="${p}-gap" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="8" height="8" fill="color-mix(in oklab, var(--rust) 16%, var(--paper))"/><line x1="0" y1="0" x2="0" y2="8" stroke="var(--rust)" stroke-width="1.6"/></pattern>
    </defs>`;
  const fillFor = (p, key) => (SERIES[key].hatch ? `url(#${p}-${SERIES[key].hatch})` : SERIES[key].color);

  // A bar with a 4px rounded data-end, square at the baseline.
  function bar(x, y, w, h, r = 4) {
    h = Math.max(h, 0);
    r = Math.min(r, h, w / 2);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }
  function hbar(x, y, w, h, r = 4) {
    w = Math.max(w, 0);
    r = Math.min(r, w, h / 2);
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  }

  const AX = `font-family="IBM Plex Mono" font-size="15" fill="var(--ink-3)" letter-spacing=".04em"`;

  /* ── Legend (HTML) ─────────────────────────────────────────────────────── */
  function legend(keys) {
    return keys
      .map((k) => {
        const s = SERIES[k];
        const sw = `<svg width="30" height="20" viewBox="0 0 30 20">${hatchDefs("lg" + k)}<rect x="1" y="1" width="28" height="18" rx="4" fill="${fillFor("lg" + k, k)}"/></svg>`;
        return `<span class="lg ${k === "ours" ? "ours" : ""}">${sw}<span>${s.name}</span></span>`;
      })
      .join("");
  }

  /* ── The leaf: a botanical line drawing, reused by the hook and the tiles ─ */
  const LESIONS = [
    [262, 150, 18], [140, 228, 13], [282, 262, 22], [178, 306, 10],
    [236, 330, 12], [122, 150, 7], [300, 200, 6], [160, 180, 5], [212, 96, 6],
  ];
  function leafSymbol() {
    const veins = [
      ["M202 110 C 225 100, 250 86, 268 70", "M202 110 C 180 100, 160 88, 140 72"],
      ["M204 160 C 240 146, 280 128, 310 104", "M204 160 C 168 146, 128 128, 100 106"],
      ["M205 215 C 245 200, 290 180, 330 152", "M205 215 C 165 200, 115 182, 76 156"],
      ["M205 270 C 245 258, 290 240, 330 214", "M205 270 C 165 258, 118 242, 74 218"],
      ["M205 322 C 238 314, 272 300, 300 280", "M205 322 C 172 314, 138 302, 108 284"],
    ].flat();
    const lesions = LESIONS.map(
      ([x, y, r]) => `
        <circle cx="${x}" cy="${y}" r="${r * 1.75}" fill="#e7c95a" opacity=".55"/>
        <circle cx="${x}" cy="${y}" r="${r}" fill="#7d4426"/>
        <circle cx="${x}" cy="${y}" r="${r * 0.62}" fill="none" stroke="#c99a62" stroke-width="${Math.max(0.8, r / 9)}"/>
        <circle cx="${x}" cy="${y}" r="${r * 0.26}" fill="#5a2f19"/>`
    ).join("");
    return `
      <symbol id="leafsym" viewBox="0 0 400 400">
        <defs><clipPath id="leafclip"><path d="M200 28 C 258 62, 338 128, 342 218 C 346 298, 284 360, 204 372 C 126 362, 58 300, 60 218 C 62 130, 140 62, 200 28 Z"/></clipPath></defs>
        <path d="M204 372 C 206 384, 212 396, 226 406" fill="none" stroke="#14120c" stroke-width="4" stroke-linecap="round"/>
        <path d="M200 28 C 258 62, 338 128, 342 218 C 346 298, 284 360, 204 372 C 126 362, 58 300, 60 218 C 62 130, 140 62, 200 28 Z" fill="#dfe7c8"/>
        <g clip-path="url(#leafclip)">
          <g fill="none" stroke="#5c7a45" stroke-width="2" stroke-linecap="round" opacity=".75">${veins.map((d) => `<path d="${d}"/>`).join("")}</g>
          ${lesions}
        </g>
        <path d="M200 40 C 206 130, 206 260, 204 370" fill="none" stroke="#3f5a2e" stroke-width="3" stroke-linecap="round"/>
        <path d="M200 28 C 258 62, 338 128, 342 218 C 346 298, 284 360, 204 372 C 126 362, 58 300, 60 218 C 62 130, 140 62, 200 28 Z" fill="none" stroke="#14120c" stroke-width="3.2" stroke-linejoin="round"/>
      </symbol>`;
  }

  // Maize blade with elongated, wavy-margined tan lesions (the NCLB example).
  function maizeBlade() {
    const outline = "M18 70 C 200 36, 520 26, 720 48 C 770 54, 800 62, 822 70 C 800 78, 770 86, 720 92 C 520 114, 200 106, 18 96 Z";
    const veins = [56, 63, 77, 84].map((y) => `<path d="M30 ${y + (y < 70 ? 4 : -4)} C 260 ${y - 8}, 560 ${y - 6}, 800 ${70 + (y - 70) * 0.15}" />`).join("");
    const lesions = [
      [170, 60, 62, 9, -3], [330, 82, 80, 10, 2], [470, 58, 54, 8, -2], [580, 80, 44, 7, 1], [255, 86, 30, 6, 2], [660, 64, 30, 6, -2],
    ].map(([x, y, rx, ry, a]) => `
      <ellipse cx="${x}" cy="${y}" rx="${rx + 9}" ry="${ry + 5}" fill="#e7c95a" opacity=".5" transform="rotate(${a} ${x} ${y})"/>
      <ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="#c9a274" stroke="#7d4426" stroke-width="1.6" transform="rotate(${a} ${x} ${y})"/>`).join("");
    return `<svg viewBox="0 0 840 140" class="maize" aria-label="Maize leaf with elongated tan lesions">
      <defs><clipPath id="mzclip"><path d="${outline}"/></clipPath></defs>
      <path d="${outline}" fill="#dfe7c8"/>
      <g clip-path="url(#mzclip)"><g fill="none" stroke="#5c7a45" stroke-width="1.4" opacity=".6">${veins}</g>${lesions}</g>
      <path d="M22 83 C 260 70, 560 66, 818 70" fill="none" stroke="#3f5a2e" stroke-width="2.6"/>
      <path d="${outline}" fill="none" stroke="#14120c" stroke-width="2.6" stroke-linejoin="round"/>
    </svg>`;
  }

  // A crop of the leaf, as an AnyRes tile sees it.
  function leafTile(vb) {
    return `<svg viewBox="${vb}" preserveAspectRatio="xMidYMid slice"><use href="#leafsym" width="400" height="400"/></svg>`;
  }

  /* ── Slide: compression: 3,645 dots fold into 128 ─────────────────────── */
  function tokens(el) {
    const W = 880, H = 610, per = 27, sp = 5.5, block = per * sp, gap = (W - 5 * block) / 4;
    const tiles = ["GL", "TL", "TR", "BL", "BR"];
    let dots = "", labels = "", fans = "";
    const gx = 280, gy = 410, cols = 16, rows = 8, gs = 20;
    const gridW = (cols - 1) * gs, gridCX = gx + gridW / 2;
    tiles.forEach((t, b) => {
      const x0 = b * (block + gap);
      let pts = "";
      for (let i = 0; i < per; i++) for (let j = 0; j < per; j++) pts += `M${(x0 + j * sp + sp / 2).toFixed(1)},${(52 + i * sp + sp / 2).toFixed(1)}h0`;
      dots += `<path d="${pts}" stroke="var(--ink-2)" stroke-width="2.6" stroke-linecap="round" opacity=".7"/>`;
      labels += `<text x="${x0 + block / 2}" y="34" text-anchor="middle" ${AX}>${t}</text>`;
      for (let k = 0; k < 4; k++) {
        const sx = x0 + block * (0.18 + k * 0.21), sy = 52 + block + 6;
        const ex = gridCX + (sx - W / 2) * 0.32, ey = gy - 22;
        fans += `<path class="draw" pathLength="1" style="--d:${200 + b * 60}ms" d="M${sx},${sy} C ${sx},${sy + 120} ${ex},${ey - 120} ${ex},${ey}" fill="none" stroke="var(--leaf)" stroke-width="1.6" opacity=".55"/>`;
      }
    });
    let lat = "";
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      lat += `<circle class="pop" style="--i:${i};--d:900ms" cx="${gx + c * gs}" cy="${gy + r * gs}" r="6.2" fill="var(--leaf)"/>`;
    }
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
      ${labels}${dots}${fans}
      <rect x="${gx - 22}" y="${gy - 22}" width="${gridW + 44}" height="${(rows - 1) * gs + 44}" rx="16" fill="none" stroke="var(--leaf-deep)" stroke-width="1.5" stroke-dasharray="5 6"/>
      ${lat}
      <text x="${gx - 44}" y="${gy + 60}" text-anchor="end" font-family="Cormorant Garamond" font-size="44" font-weight="600" fill="var(--ink)">128</text>
      <text x="${gx - 44}" y="${gy + 88}" text-anchor="end" ${AX}>LEARNED LATENTS</text>
      <text x="${gx + gridW + 44}" y="${gy + 60}" ${AX}>128 × 3072</text>
      <rect x="${W / 2 - 215}" y="${gy - 121}" width="430" height="42" rx="21" fill="var(--paper)" stroke="var(--rule)" class="fade" style="--d:700ms"/>
      <text x="${W / 2}" y="${gy - 92}" text-anchor="middle" font-family="Newsreader" font-style="italic" font-size="24" fill="var(--ink-2)">cross-attention · 2 Perceiver blocks</text>
    </svg>`;
  }

  /* ── Slide: compression: the quadratic and the line ───────────────────── */
  function flops(el) {
    const { perTile, N, d } = AP.complexity;
    const full = (k) => (4 * (perTile * k) ** 2 * d) / 1e9;
    const perc = (k) => (4 * N * perTile * k * d + 4 * N * N * d) / 1e9;
    const W = 760, H = 600, m = { l: 92, r: 40, t: 40, b: 78 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = (k) => m.l + ((k - 1) / 9) * pw, Y = (v) => m.t + ph - (v / 700) * ph;
    const path = (f) => { let s = ""; for (let k = 1; k <= 10.0001; k += 0.1) s += `${s ? "L" : "M"}${X(k).toFixed(1)},${Y(f(k)).toFixed(1)}`; return s; };
    let grid = "";
    [0, 200, 400, 600].forEach((v) => {
      grid += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule-soft)"/>
               <text x="${m.l - 14}" y="${Y(v) + 5}" text-anchor="end" ${AX}>${v}</text>`;
    });
    for (let k = 1; k <= 10; k++) grid += `<text x="${X(k)}" y="${H - m.b + 30}" text-anchor="middle" ${AX}>${k}</text>`;
    const mark = (k, ratio, at) => {
      const lx = X(k) + at.dx, ly = Y(at.v), anchor = at.dx > 0 ? "start" : "end";
      const a = full(k), b = perc(k);
      return `<g class="fade" style="--d:${k === 5 ? 1300 : 1700}ms">
        <line x1="${X(k)}" x2="${X(k)}" y1="${Y(a)}" y2="${Y(b)}" stroke="var(--ink)" stroke-dasharray="3 5" stroke-width="1.4"/>
        <circle cx="${X(k)}" cy="${Y(a)}" r="8" fill="var(--paper)" stroke="var(--rust)" stroke-width="3" data-tip="Full self-attention · ${k} tiles\n${a.toFixed(1)} GFLOP / layer"/>
        <circle cx="${X(k)}" cy="${Y(b)}" r="8" fill="var(--paper)" stroke="var(--leaf)" stroke-width="3" data-tip="Perceiver · ${k} tiles\n${b.toFixed(2)} GFLOP / layer"/>
        <text x="${lx}" y="${ly}" text-anchor="${anchor}" stroke="var(--field)" stroke-width="7" paint-order="stroke" stroke-linejoin="round" font-family="Cormorant Garamond" font-weight="600" font-size="46" fill="var(--ink)">${ratio}</text>
        <text x="${lx}" y="${ly + 26}" text-anchor="${anchor}" ${AX} stroke="var(--field)" stroke-width="6" paint-order="stroke" stroke-linejoin="round">${a.toFixed(1)} → ${b.toFixed(2)} G</text>
      </g>`;
    };
    // A ×50 zoom of the Perceiver alone. It is a separate small chart with its
    // own axis, never a second scale on the main plot.
    function zoom() {
      const x0 = 104, y0 = 54, w = 230, h = 190, il = x0 + 46, ir = x0 + w - 18, it = y0 + 44, ib = y0 + h - 30;
      const zx = (k) => il + ((k - 1) / 9) * (ir - il), zy = (v) => ib - (v / 14) * (ib - it);
      let zp = ""; for (let k = 1; k <= 10.0001; k += 0.25) zp += `${zp ? "L" : "M"}${zx(k).toFixed(1)},${zy(perc(k)).toFixed(1)}`;
      const ZT = `font-family="IBM Plex Mono" font-size="13" fill="var(--ink-3)"`;
      const lead = x0 + w / 2;
      return `<g class="fade" style="--d:2100ms">
        <line x1="${lead}" x2="${lead}" y1="${y0 + h}" y2="${Y(17)}" stroke="var(--leaf-deep)" stroke-width="1.6" stroke-dasharray="2 5"/>
        <rect x="${x0}" y="${y0}" width="${w}" height="${h}" rx="14" fill="var(--paper)" stroke="var(--leaf-deep)" stroke-width="1.6"/>
        <text x="${x0 + 16}" y="${y0 + 26}" font-family="IBM Plex Mono" font-size="13" fill="var(--ink)" letter-spacing=".08em">PERCEIVER · ×50 ZOOM</text>
        ${[0, 6, 12].map((v) => `<line x1="${il}" x2="${ir}" y1="${zy(v)}" y2="${zy(v)}" stroke="var(--rule-soft)"/><text x="${il - 8}" y="${zy(v) + 4}" text-anchor="end" ${ZT}>${v}</text>`).join("")}
        <line x1="${il}" x2="${ir}" y1="${zy(0)}" y2="${zy(0)}" stroke="var(--ink)" stroke-width="1.2"/>
        ${[1, 5, 10].map((k) => `<text x="${zx(k)}" y="${ib + 18}" text-anchor="middle" ${ZT}>${k}</text>`).join("")}
        <path d="${zp}" fill="none" stroke="var(--leaf)" stroke-width="3.5" stroke-linecap="round"/>
        <circle cx="${zx(5)}" cy="${zy(perc(5))}" r="5.5" fill="var(--paper)" stroke="var(--leaf)" stroke-width="2.5"/>
        <circle cx="${zx(10)}" cy="${zy(perc(10))}" r="5.5" fill="var(--paper)" stroke="var(--leaf)" stroke-width="2.5"/>
        <text x="${zx(5) - 10}" y="${zy(perc(5)) - 8}" text-anchor="end" font-family="IBM Plex Mono" font-size="15" font-weight="500" fill="var(--ink)">5.93 G</text>
        <text x="${zx(10) - 4}" y="${zy(perc(10)) - 12}" text-anchor="end" font-family="IBM Plex Mono" font-size="15" font-weight="500" fill="var(--ink)">11.7 G</text>
      </g>`;
    }
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
      <rect x="${X(4.6)}" y="${m.t}" width="${X(5.4) - X(4.6)}" height="${ph}" fill="color-mix(in oklab, var(--field) 45%, transparent)" rx="8"/>
      <text x="${X(5)}" y="${m.t - 12}" text-anchor="middle" ${AX}>DEPLOYED</text>
      ${grid}
      <line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="1.5"/>
      <path class="draw" pathLength="1" d="${path(full)}" fill="none" stroke="var(--rust)" stroke-width="4" stroke-linecap="round"/>
      <rect class="fade" style="--d:1900ms" x="${X(1) - 10}" y="${Y(17)}" width="${X(10) - X(1) + 20}" height="${Y(0) - Y(17) + 10}" rx="10" fill="color-mix(in oklab, var(--leaf) 10%, transparent)" stroke="var(--leaf-deep)" stroke-width="1.6" stroke-dasharray="5 5"/>
      <path class="draw" pathLength="1" style="--d:250ms" d="${path(perc)}" fill="none" stroke="var(--leaf)" stroke-opacity=".28" stroke-width="14" stroke-linecap="round"/>
      <path class="draw" pathLength="1" style="--d:250ms" d="${path(perc)}" fill="none" stroke="var(--leaf)" stroke-width="5.5" stroke-linecap="round"/>
      ${mark(5, "27.5×", { v: 228, dx: -16 })}${mark(10, "56×", { v: 300, dx: -16 })}
      <text x="${X(9.45)}" y="${Y(625)}" text-anchor="end" font-family="Newsreader" font-style="italic" font-size="25" fill="var(--ink)">full self-attention · O(T²)</text>
      ${zoom()}
      <text x="${X(9.6)}" y="${Y(perc(9.6)) - 18}" text-anchor="end" font-family="Newsreader" font-style="italic" font-size="25" fill="var(--ink)">Perceiver · O(N·T)</text>
      <text x="${m.l + pw / 2}" y="${H - 8}" text-anchor="middle" ${AX}>ANYRES TILES  (729 TOKENS EACH)</text>
      <text transform="translate(22 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle" ${AX}>ATTENTION GFLOP / LAYER</text>
    </svg>`;
  }

  /* ── Slide: data: a Sankey of where the 117,635 images went ───────────── */
  function flow(el) {
    const D = AP.data, W = 1040, H = 640, top = 150, span = 380;
    const s = span / D.images;
    const xA = 0, xB = 380, xC = 760, nw = 26;
    const hKept = D.kept * s, hDrop = (D.images - D.kept) * s, hTrain = D.train * s, hTest = D.test * s;
    const aTop = top, bTop = top, dropTop = top + hKept + 60, cTop = top - 10, testTop = cTop + hTrain + 46;
    const ribbon = (x0, y0a, y0b, x1, y1a, y1b, fill, cls = "", st = "") => {
      const mx = (x0 + x1) / 2;
      return `<path class="${cls}" style="${st}" d="M${x0},${y0a} C${mx},${y0a} ${mx},${y1a} ${x1},${y1a} L${x1},${y1b} C${mx},${y1b} ${mx},${y0b} ${x0},${y0b} Z" fill="${fill}"/>`;
    };
    let buckets = "", y = cTop, bl = "";
    D.buckets.forEach((b, i) => {
      const h = hTrain * b.share;
      const op = [1, 0.7, 0.42][i];
      buckets += `<rect x="${xC}" y="${y}" width="${nw}" height="${h}" fill="var(--leaf)" opacity="${op}" data-tip="${b.name} bucket · confidence ${b.range}\nweight w = ${b.w} · ~${Math.round(b.share * 100)}% of training"/>`;
      bl += `<text x="${xC + nw + 16}" y="${y + h / 2 + 6}" font-family="IBM Plex Mono" font-size="16" fill="var(--ink-2)">${b.name} · w=${b.w.toFixed(1)} · ${Math.round(b.share * 100)}%</text>`;
      y += h;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
      <g class="fade" style="--d:300ms">
        ${ribbon(xA + nw, aTop, aTop + hKept, xB, bTop, bTop + hKept, "color-mix(in oklab, var(--leaf) 22%, transparent)")}
        ${ribbon(xA + nw, aTop + hKept, aTop + hKept + hDrop, xB, dropTop, dropTop + hDrop, "color-mix(in oklab, var(--rust) 22%, transparent)")}
      </g>
      <g class="fade" style="--d:700ms">
        ${ribbon(xB + nw, bTop, bTop + hTrain, xC, cTop, cTop + hTrain, "color-mix(in oklab, var(--leaf) 22%, transparent)")}
        ${ribbon(xB + nw, bTop + hTrain, bTop + hKept, xC, testTop, testTop + hTest, "color-mix(in oklab, var(--ink) 22%, transparent)")}
      </g>
      <rect x="${xA}" y="${aTop}" width="${nw}" height="${span}" rx="3" fill="var(--ink)" data-tip="117,635 agricultural leaf images"/>
      <rect x="${xB}" y="${bTop}" width="${nw}" height="${hKept}" rx="3" fill="var(--ink)" data-tip="101,301 kept (86%)\nteacher confidence ≥ τ = 0.5"/>
      <rect x="${xB}" y="${dropTop}" width="${nw}" height="${hDrop}" rx="3" fill="var(--rust)" data-tip="16,334 filtered out (14%)"/>
      ${buckets}
      <rect x="${xC}" y="${testTop}" width="${nw}" height="${Math.max(hTest, 4)}" rx="2" fill="var(--ink)" data-tip="5,062 held-out test samples"/>

      <text x="${xA}" y="${aTop - 62}" font-family="Cormorant Garamond" font-weight="600" font-size="58" fill="var(--ink)">117,635</text>
      <text x="${xA}" y="${aTop - 28}" ${AX}>LEAF IMAGES</text>
      <text x="${xB}" y="${aTop - 62}" font-family="Cormorant Garamond" font-weight="600" font-size="58" fill="var(--ink)">101,301</text>
      <text x="${xB}" y="${aTop - 28}" ${AX}>KEPT · 86%</text>
      <text x="${xB + nw + 16}" y="${dropTop + hDrop / 2 + 6}" font-family="IBM Plex Mono" font-size="16" fill="var(--rust)">16,334 dropped · low teacher confidence</text>
      <text x="${xC}" y="${cTop - 62}" font-family="Cormorant Garamond" font-weight="600" font-size="58" fill="var(--ink)">96,239</text>
      <text x="${xC}" y="${cTop - 28}" ${AX}>TRAIN · WEIGHTED</text>
      ${bl}
      <text x="${xC + nw + 16}" y="${testTop + 6}" font-family="IBM Plex Mono" font-size="16" fill="var(--ink)">5,062 held-out test</text>

      <g transform="translate(${(xA + nw + xB) / 2} ${top + hKept / 2})">
        <rect x="-150" y="-50" width="300" height="100" rx="16" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
        <text y="-12" text-anchor="middle" font-family="Cormorant Garamond" font-weight="600" font-size="30" fill="var(--ink)">Gemma-3-12B-IT</text>
        <text y="20" text-anchor="middle" ${AX}>TEACHER · 4-BIT · 8-SHOT</text>
      </g>
      <g transform="translate(${(xB + nw + xC) / 2} ${top + hKept / 2 - 12})">
        <rect x="-140" y="-46" width="280" height="92" rx="16" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
        <text y="-8" text-anchor="middle" font-family="Cormorant Garamond" font-weight="600" font-size="28" fill="var(--ink)">filter at τ = 0.5</text>
        <text y="22" text-anchor="middle" ${AX}>TOKEN-PROBABILITY CONF.</text>
      </g>
    </svg>`;
  }

  /* ── Slide: per-class F1 ───────────────────────────────────────────────── */
  function perClass(el) {
    const keys = ["llava", "intern", "qwen", "ours"], C = AP.classes;
    const W = 1696, H = 540, m = { l: 74, r: 10, t: 34, b: 92 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b, gw = pw / C.length, bw = 50, bg = 4;
    const groupW = keys.length * bw + (keys.length - 1) * bg;
    const Y = (v) => m.t + ph - v * ph;
    let g = "";
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach((v) => {
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule-soft)"/><text x="${m.l - 14}" y="${Y(v) + 5}" text-anchor="end" ${AX}>${v.toFixed(1)}</text>`;
    });
    let bars = "", i = 0;
    C.forEach((c, ci) => {
      const gx = m.l + ci * gw + (gw - groupW) / 2;
      keys.forEach((k, ki) => {
        const v = AP.perClassF1[k][c];
        const x = gx + ki * (bw + bg), h = Math.max(v * ph, 2);
        bars += `<path class="grow-y" style="--i:${i++}" d="${bar(x, Y(0) - h, bw, h)}" fill="${fillFor("pc", k)}" data-tip="${SERIES[k].name} · ${c}\nF1 ${fmt(v)}"/>`;
        if (k === "ours") bars += `<text class="fade" style="--d:1100ms" x="${x + bw / 2}" y="${Y(v) - 12}" text-anchor="middle" font-family="IBM Plex Mono" font-weight="500" font-size="19" fill="var(--leaf-deep)" stroke="var(--field)" stroke-width="6" paint-order="stroke" stroke-linejoin="round">${fmt(v, 2)}</text>`;
      });
      g += `<text x="${m.l + ci * gw + gw / 2}" y="${H - m.b + 40}" text-anchor="middle" font-family="Newsreader" font-size="28" fill="var(--ink)">${c}</text>
            <text x="${m.l + ci * gw + gw / 2}" y="${H - m.b + 68}" text-anchor="middle" ${AX}>n = ${thousands(AP.support[c])}</text>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${hatchDefs("pc")}${g}
      <line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="1.5"/>${bars}
      <text transform="translate(18 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle" ${AX}>PATHOLOGY-TYPE F1</text></svg>`;
  }

  /* ── Slide: confusion matrix (row-normalised) ──────────────────────────── */
  function confusion(el) {
    const M = AP.confusion, L = AP.cmShort, n = M.length, cs = 96, gp = 6, ox = 120, oy = 76;
    const callouts = { "5,1": true, "3,2": true, "0,2": true, "4,1": true };
    let cells = "", k = 0;
    M.forEach((row, r) => {
      const tot = row.reduce((a, b) => a + b, 0);
      row.forEach((v, c) => {
        const p = v / tot, x = ox + c * (cs + gp), y = oy + r * (cs + gp);
        const diag = r === c, hot = callouts[`${r},${c}`];
        const fill = `color-mix(in oklab, var(--leaf-deep) ${Math.round(Math.min(1, p * 1.15) * 100)}%, var(--paper))`;
        cells += `<g class="pop" style="--i:${k++ * 12}">
          <rect x="${x}" y="${y}" width="${cs}" height="${cs}" rx="12" fill="${fill}" ${hot ? `stroke="var(--rust)" stroke-width="4"` : `stroke="var(--rule-soft)"`}
            data-tip="true ${AP.cmLabels[r]} → predicted ${AP.cmLabels[c]}\n${v} of ${thousands(tot)} · ${(p * 100).toFixed(1)}%"/>
          <text x="${x + cs / 2}" y="${y + cs / 2 + 10}" text-anchor="middle" font-family="IBM Plex Mono" font-size="${diag ? 28 : 24}" font-weight="${diag || hot ? 500 : 400}"
            fill="${p > 0.45 ? "var(--paper)" : hot ? "var(--rust)" : "var(--ink-2)"}" pointer-events="none">${Math.round(p * 100)}</text></g>`;
      });
    });
    let axes = "";
    L.forEach((l, i) => {
      axes += `<text x="${ox - 18}" y="${oy + i * (cs + gp) + cs / 2 + 6}" text-anchor="end" font-family="IBM Plex Mono" font-size="19" fill="var(--ink-2)">${l}</text>`;
      axes += `<text x="${ox + i * (cs + gp) + cs / 2}" y="${oy - 16}" text-anchor="middle" font-family="IBM Plex Mono" font-size="19" fill="var(--ink-2)">${l}</text>`;
    });
    const span = n * (cs + gp) - gp;
    el.innerHTML = `<svg viewBox="0 0 ${ox + span + 10} ${oy + span + 60}" width="${ox + span + 10}" height="${oy + span + 60}">
      <text x="${ox + span / 2}" y="20" text-anchor="middle" ${AX}>PREDICTED →</text>
      <text transform="translate(26 ${oy + span / 2}) rotate(-90)" text-anchor="middle" ${AX}>TRUE CLASS →</text>
      ${axes}${cells}
      <text x="${ox}" y="${oy + span + 44}" ${AX}>ROW %  ·  n = 5,062</text></svg>`;
  }

  /* ── Slide: calibration: reliability diagram ──────────────────────────── */
  function reliability(el) {
    // Only 5 of the 10 bins hold predictions, so the bins are drawn side by
    // side: a continuous axis would show the empty 0.5–0.6 bin as a hole.
    const B = AP.calibration.bins, W = 900, H = 660, m = { l: 92, r: 24, t: 96, b: 120 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b, band = pw / B.length, bw = 96;
    const Y = (v) => m.t + ph - v * ph;
    // The bin that adds most to ECE: its share of samples times its gap.
    const worst = B.reduce((a, b, i) => (b.n * (b.conf - b.acc) > B[a].n * (B[a].conf - B[a].acc) ? i : a), 0);
    let g = "";
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule-soft)"/><text x="${m.l - 14}" y="${Y(v) + 5}" text-anchor="end" ${AX}>${v.toFixed(2)}</text>`;
    });
    let marks = "";
    B.forEach((b, i) => {
      const cx = m.l + band * (i + 0.5), x = cx - bw / 2, sparse = b.n < 100;
      if (i === worst) g += `<rect x="${cx - band / 2 + 6}" y="${m.t - 12}" width="${band - 12}" height="${ph + 12}" rx="10" fill="color-mix(in oklab, var(--field) 55%, transparent)"/>
        <text x="${cx}" y="${m.t + 8}" text-anchor="middle" ${AX} stroke="var(--field)" stroke-width="6" paint-order="stroke" stroke-linejoin="round">LARGEST GAP</text>`;
      marks += `<g opacity="${sparse ? 0.55 : 1}">
        <path class="grow-y" style="--i:${i * 3}" d="${bar(x, Y(b.acc), bw, Y(0) - Y(b.acc))}" fill="var(--leaf)"
          data-tip="confidence ${b.lo.toFixed(1)}–${b.hi.toFixed(1)} · n = ${thousands(b.n)}\nright ${b.acc.toFixed(3)} of the time"/>
        <rect class="fade" style="--d:${1000 + i * 120}ms" x="${x}" y="${Y(b.conf)}" width="${bw}" height="${Y(b.acc) - Y(b.conf)}" fill="url(#rl-gap)" stroke="var(--rust)" stroke-width="1.5"
          data-tip="overconfident by ${(b.conf - b.acc).toFixed(3)}"/>
        <line class="fade" style="--d:${1100 + i * 120}ms" x1="${x - 12}" x2="${x + bw + 12}" y1="${Y(b.conf)}" y2="${Y(b.conf)}" stroke="var(--ink)" stroke-width="4" stroke-linecap="round"
          data-tip="said ${b.conf.toFixed(3)} on average"/>
        <text class="fade" style="--d:1400ms" x="${cx}" y="${Y(b.conf) - 14}" text-anchor="middle" ${AX} stroke="var(--field)" stroke-width="6" paint-order="stroke" stroke-linejoin="round">n = ${thousands(b.n)}</text>
      </g>
      <text x="${cx}" y="${Y(0) + 32}" text-anchor="middle" font-family="IBM Plex Mono" font-size="18" fill="var(--ink-2)">${b.lo.toFixed(1)}–${b.hi.toFixed(1)}</text>`;
    });
    const L = `font-family="Newsreader" font-size="20" fill="var(--ink-2)"`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${hatchDefs("rl")}
      <rect x="${m.l}" y="22" width="24" height="16" rx="4" fill="var(--leaf)"/><text x="${m.l + 34}" y="36" ${L}>accuracy: what happened</text>
      <line x1="362" x2="390" y1="30" y2="30" stroke="var(--ink)" stroke-width="4" stroke-linecap="round"/><text x="400" y="36" ${L}>confidence: what it said</text>
      <rect x="652" y="22" width="24" height="16" rx="3" fill="url(#rl-gap)" stroke="var(--rust)" stroke-width="1.5"/><text x="686" y="36" ${L}>the gap</text>
      ${g}
      <line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="1.5"/>
      ${marks}
      <text x="${m.l + pw / 2}" y="${Y(0) + 64}" text-anchor="middle" ${AX}>STATED-CONFIDENCE BIN</text>
      <text x="${m.l + pw / 2}" y="${Y(0) + 98}" text-anchor="middle" font-family="Newsreader" font-style="italic" font-size="19" fill="var(--ink-3)">Only 5 of 10 bins hold predictions: none below 0.4, none between 0.5 and 0.6.</text>
      <text transform="translate(22 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle" ${AX}>SCORE (0–1)</text></svg>`;
  }

  /* ── Backup: where the parameters live ─────────────────────────────────── */
  function params(el) {
    const P = AP.params, W = 1696, H = 150, total = P.siglip + P.bridge + P.phi + P.lora, s = W / total;
    const segs = [
      { k: "SigLIP-SO400M", v: P.siglip, frozen: true },
      { k: "Bridge", v: P.bridge, frozen: false },
      { k: "Phi-3-mini-128k", v: P.phi, frozen: true },
      { k: "LoRA", v: P.lora, frozen: false },
    ];
    let x = 0, out = "", lab = "";
    segs.forEach((g, i) => {
      const w = g.v * s;
      out += `<rect class="grow-x" style="--i:${i * 4}" x="${x + 1}" y="40" width="${Math.max(w - 2, 3)}" height="64" rx="8"
        fill="${g.frozen ? "url(#pp-frz)" : "var(--leaf)"}" stroke="${g.frozen ? "var(--ink-3)" : "var(--leaf-deep)"}" stroke-width="1.5"
        data-tip="${g.k} · ${g.v >= 1000 ? (g.v / 1000).toFixed(2) + "B" : Math.round(g.v) + "M"} · ${g.frozen ? "frozen" : "trained"}"/>`;
      const anchor = i === 3 ? "end" : "start";
      const tx = i === 3 ? x + w : x + 4;
      lab += `<text x="${tx}" y="${i % 2 ? 136 : 26}" text-anchor="${anchor}" font-family="IBM Plex Mono" font-size="16" fill="${g.frozen ? "var(--ink-3)" : "var(--leaf-deep)"}">${g.k} · ${g.v >= 1000 ? (g.v / 1000).toFixed(2) + "B" : Math.round(g.v) + "M"}</text>`;
      x += w;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
      <defs><pattern id="pp-frz" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="10" height="10" fill="var(--paper-sunk)"/><line x1="0" y1="0" x2="0" y2="10" stroke="rgba(20,18,12,.12)" stroke-width="2"/></pattern></defs>
      ${out}${lab}</svg>`;
  }

  /* ── Slide: the road ahead: one smooth wave, milestones on its crests ─── */
  function roadmap(el, items) {
    const W = 1696, H = 640, n = items.length, x0 = 170, x1 = 1480, crest = 270, trough = 370;
    const pts = items.map((_, i) => [x0 + (i * (x1 - x0)) / (n - 1), i % 2 ? trough : crest]);
    let d = `M0,${(crest + trough) / 2 + 30} C ${pts[0][0] * 0.5},${(crest + trough) / 2 + 30} ${pts[0][0] * 0.5},${pts[0][1]} ${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < n; i++) {
      const [ax, ay] = pts[i - 1], [bx, by] = pts[i], mx = (ax + bx) / 2;
      d += ` C ${mx},${ay} ${mx},${by} ${bx},${by}`;
    }
    const [lx, ly] = pts[n - 1];
    d += ` C ${lx + 70},${ly} ${lx + 70},${crest - 70} ${W - 30},${crest - 70}`;
    let nodes = "", cards = "";
    pts.forEach(([x, y], i) => {
      const it = items[i];
      nodes += `<g class="pop" style="--d:${500 + i * 230}ms"><circle cx="${x}" cy="${y}" r="24" fill="${it.tone === "now" ? "var(--leaf)" : "var(--paper)"}" stroke="var(--ink)" stroke-width="2"/>
        <text x="${x}" y="${y + 7}" text-anchor="middle" font-family="IBM Plex Mono" font-size="19" font-weight="500" fill="${it.tone === "now" ? "var(--paper)" : "var(--ink)"}">${String(i + 1).padStart(2, "0")}</text></g>`;
      const above = i % 2 === 0;
      cards += `<div class="rm-card r" style="--d:${400 + i * 230}ms;left:${x - 165}px;${above ? `bottom:${H - y + 44}px` : `top:${y + 44}px`}">
        <span class="label ${it.tone === "now" ? "leaf-ink" : ""}">${it.when}</span>
        <h4>${it.title}</h4><p>${it.body}</p></div>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="position:absolute;inset:0">
        <path class="draw" pathLength="1" d="${d}" fill="none" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M${W - 46},${crest - 82} L${W - 28},${crest - 70} L${W - 46},${crest - 58}" fill="none" stroke="var(--ink)" stroke-width="2.5" class="fade" style="--d:1500ms"/>
        ${nodes}</svg>${cards}`;
  }

  window.Charts = { leafSymbol, leafTile, maizeBlade, tokens, flops, flow, perClass, confusion, reliability, params, roadmap, legend };
})();
