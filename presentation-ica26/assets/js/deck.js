/* The deck engine. No framework: slides are <section class="slide"> authored
   at 1920 × 1080; this file scales them, moves between them, plays the builds,
   and keeps the presenter view in sync over a BroadcastChannel.

   URL modes
     index.html#/6/2      slide 6, build step 2
     ?preview             an embedded, non-interactive view (presenter thumbnails)
     ?static              every animation at its final frame
     ?print               static + print layout (used by tools/export.mjs)        */

(function () {
  const params = new URLSearchParams(location.search);
  const PREVIEW = params.has("preview");
  const STATIC = params.has("static") || params.has("print");
  const body = document.body;
  if (PREVIEW) body.classList.add("preview");
  if (STATIC) body.classList.add("static");

  const stage = document.querySelector(".stage");
  const slides = Array.from(document.querySelectorAll(".slide"));
  const total = slides.length;
  const mainCount = slides.filter((s) => !s.hasAttribute("data-backup")).length;
  const channel = "BroadcastChannel" in window ? new BroadcastChannel("agriperceiver-ica26") : null;
  let idx = 0, step = 0;

  /* ── Charts first, so builds and stagger see the generated marks ─────── */
  const sym = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sym.setAttribute("width", "0"); sym.setAttribute("height", "0");
  sym.style.position = "absolute";
  sym.innerHTML = Charts.leafSymbol();
  body.prepend(sym);
  document.querySelectorAll("[data-chart]").forEach((el) => {
    const name = el.dataset.chart;
    if (name === "roadmap") Charts.roadmap(el, window.ROADMAP);
    else if (Charts[name]) Charts[name](el);
  });
  document.querySelectorAll("[data-tile]").forEach((el) => (el.innerHTML = Charts.leafTile(el.dataset.tile)));
  document.querySelectorAll("[data-legend]").forEach((el) => (el.innerHTML = Charts.legend(el.dataset.legend.split(","))));
  document.querySelectorAll("[data-maize]").forEach((el) => (el.innerHTML = Charts.maizeBlade()));

  /* ── Chrome: brand, section, count and the axis, on every slide ──────── */
  slides.forEach((s, i) => {
    if (s.hasAttribute("data-bare")) return;
    const n = String(i + 1).padStart(2, "0");
    const count = s.hasAttribute("data-backup") ? "B1" : `${n} / ${String(mainCount).padStart(2, "0")}`;
    const top = document.createElement("header");
    top.className = "chrome-top";
    top.innerHTML = `
      <div class="brand"><span class="brand-name">AgriPerceiver</span><span class="label">ICA 2026</span></div>
      <div class="where"><span class="label ink">${s.dataset.section || ""}</span><span class="readout">${count}</span></div>`;
    const bottom = document.createElement("footer");
    bottom.className = "chrome-bottom";
    let ticks = "";
    for (let k = 0; k < mainCount; k++) {
      const x = mainCount === 1 ? 0 : (k / (mainCount - 1)) * 100;
      ticks += `<i class="${k < i ? "past" : k === i ? "now" : ""}" style="left:${x}%"></i>`;
    }
    const fill = Math.min(i, mainCount - 1) / (mainCount - 1) * 100;
    bottom.innerHTML = `
      <div class="meta"><span class="label">Khanna · Saini · Singh · Kwatra</span><span class="label">NIT Jalandhar · Annam.ai, IIT Ropar</span></div>
      <div class="axis"><b style="width:${fill}%"></b>${ticks}</div>`;
    s.prepend(top);
    s.append(bottom);
  });

  /* ── Stagger: every .r gets --i in reading order, per slide ───────────── */
  slides.forEach((s) => {
    let i = 0;
    s.querySelectorAll(".r").forEach((el) => {
      if (!el.style.getPropertyValue("--i")) el.style.setProperty("--i", i);
      i++;
    });
  });

  const stepsIn = (s) => Math.max(0, ...Array.from(s.querySelectorAll("[data-step]")).map((e) => +e.dataset.step));

  /* ── Scale ────────────────────────────────────────────────────────────── */
  function fit() {
    if (body.classList.contains("overview")) return layoutOverview();
    const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    stage.style.setProperty("--s", s);
  }
  window.addEventListener("resize", fit);

  /* ── Count-up numbers ─────────────────────────────────────────────────── */
  function countUp(s) {
    s.querySelectorAll("[data-count]").forEach((el) => {
      const end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0);
      const fmt = (v) => (el.dataset.sep ? Math.round(v).toLocaleString("en-US") : v.toFixed(dec));
      if (STATIC) { el.textContent = fmt(end); return; }
      const t0 = performance.now(), dur = 1500, delay = +(el.dataset.delay || 250);
      el.textContent = fmt(0);
      const tick = (t) => {
        const p = Math.min(1, Math.max(0, (t - t0 - delay) / dur));
        el.textContent = fmt(end * (1 - Math.pow(2, -10 * p)) / (1 - Math.pow(2, -10)));
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = fmt(end);
      };
      requestAnimationFrame(tick);
    });
  }

  /* ── Navigation ───────────────────────────────────────────────────────── */
  function applySteps(s) {
    s.querySelectorAll("[data-step]").forEach((el) => {
      const n = +el.dataset.step;
      el.classList.toggle("on", STATIC || n <= step);
      el.classList.toggle("past-dim", !STATIC && el.hasAttribute("data-dim") && n < step);
    });
  }

  function go(i, st = 0, opts = {}) {
    i = Math.max(0, Math.min(total - 1, i));
    const changed = i !== idx || opts.force;
    const s = slides[i];
    st = Math.max(0, Math.min(stepsIn(s), st));
    if (changed) {
      slides[idx].classList.remove("is-active");
      s.classList.remove("is-active");
      void s.offsetWidth; // restart arrival animations on revisit
      s.classList.add("is-active");
      countUp(s);
    }
    idx = i;
    step = st;
    applySteps(s);
    const h = `#/${idx + 1}${step ? "/" + step : ""}`;
    if (location.hash !== h) history.replaceState(null, "", h);
    if (!PREVIEW && channel) channel.postMessage({ type: "state", idx, step, total, steps: stepsIn(s) });
  }

  function next() {
    if (step < stepsIn(slides[idx])) go(idx, step + 1);
    else if (idx < total - 1) go(idx + 1, 0);
  }
  function prev() {
    if (step > 0) go(idx, step - 1);
    else if (idx > 0) go(idx - 1, stepsIn(slides[idx - 1]));
  }

  function fromHash() {
    const m = location.hash.match(/#\/(\d+)(?:\/(\d+))?/);
    if (!m) return [0, 0];
    return [parseInt(m[1], 10) - 1, parseInt(m[2] || "0", 10)];
  }
  window.addEventListener("hashchange", () => {
    const [i, st] = fromHash();
    if (i !== idx || st !== step) go(i, st);
  });

  /* ── Overview ─────────────────────────────────────────────────────────── */
  function layoutOverview() {
    const cols = window.innerWidth > 1400 ? 4 : 3, gap = 28;
    const tw = (window.innerWidth - gap * (cols + 1)) / cols, sc = tw / 1920, th = 1080 * sc;
    slides.forEach((s, i) => {
      const c = i % cols, r = Math.floor(i / cols);
      s.style.left = `${gap + c * (tw + gap)}px`;
      s.style.top = `${gap + r * (th + gap)}px`;
      s.style.transform = `scale(${sc})`;
    });
  }
  function toggleOverview(on = !body.classList.contains("overview")) {
    body.classList.toggle("overview", on);
    if (on) layoutOverview();
    else slides.forEach((s) => { s.style.left = s.style.top = s.style.transform = ""; });
    fit();
  }
  slides.forEach((s, i) => s.addEventListener("click", () => {
    if (!body.classList.contains("overview")) return;
    toggleOverview(false);
    go(i, 0, { force: true });
  }));

  /* ── PDF ──────────────────────────────────────────────────────────────── */
  const PDF = "export/AgriPerceiver-ICA26.pdf";
  async function downloadPdf() {
    try {
      const r = await fetch(PDF, { method: "HEAD", cache: "no-store" });
      if (!r.ok) throw 0;
      const a = document.createElement("a");
      a.href = PDF;
      a.download = "AgriPerceiver-ICA26.pdf";
      document.body.append(a);
      a.click();
      a.remove();
    } catch {
      window.print(); // no pre-built file (or opened from disk): print → Save as PDF
    }
  }
  window.addEventListener("beforeprint", () => body.classList.add("static"));
  window.addEventListener("afterprint", () => { if (!STATIC) body.classList.remove("static"); });

  /* ── Presenter view ───────────────────────────────────────────────────── */
  function openPresenter() {
    window.open(`presenter.html#/${idx + 1}${step ? "/" + step : ""}`, "agriperceiver-presenter", "width=1500,height=900");
  }
  if (channel && !PREVIEW) {
    channel.onmessage = ({ data }) => {
      if (data.type === "go") go(data.idx, data.step);
      else if (data.type === "next") next();
      else if (data.type === "prev") prev();
      else if (data.type === "hello") channel.postMessage({ type: "state", idx, step, total, steps: stepsIn(slides[idx]) });
    };
  }

  /* ── Input ────────────────────────────────────────────────────────────── */
  function fullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  }
  if (!PREVIEW) {
    document.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (body.classList.contains("overview") && !["o", "O", "Escape"].includes(k)) {
        if (k === "Enter") toggleOverview(false);
        return;
      }
      if (["ArrowRight", "ArrowDown", " ", "PageDown", "Enter", "n", "N"].includes(k)) { e.preventDefault(); next(); }
      else if (["ArrowLeft", "ArrowUp", "PageUp", "Backspace", "p"].includes(k)) { e.preventDefault(); prev(); }
      else if (k === "Home") go(0);
      else if (k === "End") go(mainCount - 1);
      else if (k === "f" || k === "F") fullscreen();
      else if (k === "s" || k === "S" || k === "P") openPresenter();
      else if (k === "o" || k === "O" || (k === "Escape" && body.classList.contains("overview"))) toggleOverview();
      else if (k === "b" || k === "B" || k === "." || k === ",") body.classList.toggle("is-black");
      else if (k === "d" || k === "D") downloadPdf();
    });
    let x0 = null;
    document.addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
    document.addEventListener("touchend", (e) => {
      if (x0 === null) return;
      const dx = e.changedTouches[0].clientX - x0;
      if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
      x0 = null;
    });

    // HUD: appears with the pointer, leaves when it rests.
    // The controls come up only when the pointer goes to the bottom edge, so a
    // nudged mouse never covers the slide; an idle pointer disappears entirely.
    let hudT, idleT;
    const showHud = (ms) => {
      body.classList.add("show-hud");
      clearTimeout(hudT);
      hudT = setTimeout(() => body.classList.remove("show-hud"), ms);
    };
    document.addEventListener("mousemove", (e) => {
      body.classList.remove("idle");
      clearTimeout(idleT);
      idleT = setTimeout(() => body.classList.add("idle"), 1800);
      if (e.clientY > window.innerHeight - 90 || e.target.closest?.(".hud")) showHud(2200);
    });
    showHud(3500); // announce the controls once, on load
    document.querySelector("[data-act=presenter]")?.addEventListener("click", openPresenter);
    document.querySelector("[data-act=overview]")?.addEventListener("click", () => toggleOverview());
    document.querySelector("[data-act=full]")?.addEventListener("click", fullscreen);
    document.querySelector("[data-act=pdf]")?.addEventListener("click", downloadPdf);

    // Tooltip layer for chart marks.
    const tip = document.querySelector(".tip");
    document.addEventListener("mouseover", (e) => {
      const t = e.target.closest?.("[data-tip]");
      if (!t || !tip) return;
      tip.textContent = t.dataset.tip;
      tip.classList.add("on");
    });
    document.addEventListener("mouseout", (e) => {
      if (e.target.closest?.("[data-tip]")) tip?.classList.remove("on");
    });
    document.addEventListener("mousemove", (e) => {
      if (!tip?.classList.contains("on")) return;
      const w = tip.offsetWidth, x = e.clientX + 16 + w > window.innerWidth ? e.clientX - w - 16 : e.clientX + 16;
      tip.style.left = `${x}px`;
      tip.style.top = `${e.clientY + 18}px`;
    });
  }

  fit();
  const [i0, s0] = fromHash();
  idx = i0;
  go(i0, STATIC ? 0 : s0, { force: true });
  window.Deck = { go, next, prev, get idx() { return idx; }, get step() { return step; }, total, slides };
})();
