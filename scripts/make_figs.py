#!/usr/bin/env python
"""Generate publication figures for AgriPerceiver (ICA 2026) from REAL eval data.
All numbers trace to results/eval_results.json and paper/RESULTS.md.
Outputs vector PDFs into paper-ica26/figures/.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = r"d:\Research\agri-perceiver"
OUT = os.path.join(ROOT, "paper-ica26", "figures")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))

ev = json.load(open(os.path.join(ROOT, "results", "eval_results.json")))

# ---- global style: serif to match LNCS (CMR), embed TrueType -------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# categorical palette (validated: CVD dEmin 21.6). Ours=red (highlight).
C = {"LLaVA-NeXT-7B": "#2a78d6", "InternVL2-8B": "#1baf7a",
     "Qwen2-VL-7B": "#eda100", "AgriPerceiver": "#e34948"}
H = {"LLaVA-NeXT-7B": "////", "InternVL2-8B": "\\\\\\\\",
     "Qwen2-VL-7B": "....", "AgriPerceiver": ""}
GRID = "#dddddd"

# ==========================================================================
# FIG 1 — Per-class pathology-type F1 (grouped bars).  REAL data.
#   baselines: paper/RESULTS.md sec.2 ; ours: eval_results.json per_class f1
# ==========================================================================
classes = ["Fungal", "Unknown", "Bacterial", "Deficiency", "Pest", "Viral"]
key = {"Fungal": "fungal", "Unknown": "unknown", "Bacterial": "bacterial",
       "Deficiency": "deficiency", "Pest": "pest", "Viral": "viral"}
pc = ev["classification"]["type"]["per_class"]
ours = [pc[key[c]]["f1-score"] for c in classes]
f1 = {
    "LLaVA-NeXT-7B": {"Fungal":0.301,"Unknown":0.493,"Bacterial":0.000,"Deficiency":0.006,"Pest":0.000,"Viral":0.006},
    "InternVL2-8B":  {"Fungal":0.599,"Unknown":0.689,"Bacterial":0.003,"Deficiency":0.475,"Pest":0.227,"Viral":0.022},
    "Qwen2-VL-7B":   {"Fungal":0.637,"Unknown":0.634,"Bacterial":0.025,"Deficiency":0.386,"Pest":0.017,"Viral":0.022},
    "AgriPerceiver": {c: ours[i] for i, c in enumerate(classes)},
}
models = list(C.keys())
x = np.arange(len(classes)); w = 0.20
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for j, m in enumerate(models):
    vals = [f1[m][c] for c in classes]
    ax.bar(x + (j-1.5)*w, vals, w, label=m, color=C[m], hatch=H[m],
           edgecolor="white", linewidth=0.4)
ax.set_ylabel("Pathology-type F1")
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylim(0, 0.95)
ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right", handlelength=1.4)
save(fig, "perclass_f1")
plt.close(fig)
print("wrote perclass_f1.pdf  ours per-class:", [round(v,3) for v in ours])

# ==========================================================================
# FIG 2 — Reliability diagram (AgriPerceiver).  REAL calibration bins.
# ==========================================================================
bins = ev["calibration"]["bins"]
conf = np.array([b["avg_conf"] for b in bins])
acc = np.array([b["avg_acc"] for b in bins])
cnt = np.array([b["count"] for b in bins])
ece = ev["calibration"]["ece"]
fig, ax = plt.subplots(figsize=(3.3, 3.0))
ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=0.9, label="Perfect calibration")
# gap bars (conf - acc)
for c_, a_ in zip(conf, acc):
    ax.plot([c_, c_], [a_, c_], color="#e34948", linewidth=0.8, alpha=0.6, zorder=1)
sizes = 20 + 180 * (cnt / cnt.max())
ax.scatter(conf, acc, s=sizes, color="#2a78d6", edgecolor="white",
           linewidth=0.6, zorder=3, label="Confidence bin (area $\\propto$ count)")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Empirical accuracy")
ax.set_xlim(0.4, 1.02); ax.set_ylim(0.0, 1.02)
ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.44, 0.92, f"ECE = {ece:.3f}", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#cccccc", lw=0.5))
ax.legend(frameon=False, fontsize=6.6, loc="lower right")
save(fig, "calibration")
plt.close(fig)
print("wrote calibration.pdf  ECE=", round(ece,4))

# ==========================================================================
# FIG 3 — Row-normalized confusion matrix (AgriPerceiver).  REAL.
# ==========================================================================
cm = np.array(ev["classification"]["type"]["confusion_matrix"], dtype=float)
labels = ["Bac", "Def", "Fun", "Pst", "Unk", "Vir"]
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(3.5, 3.2))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(labels)
ax.set_yticks(range(6)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(6):
    for j in range(6):
        v = cmn[i, j]
        ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                fontsize=7.2, color="white" if v > 0.5 else "#222222")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Row fraction", fontsize=8)
cb.ax.tick_params(labelsize=7)
save(fig, "confusion")
plt.close(fig)
print("wrote confusion.pdf  diag acc=", round(np.trace(cm)/cm.sum(),4))

# ==========================================================================
# FIG 4 — Attention FLOPs vs #AnyRes tiles: full self-attn O(T^2) vs
#   Perceiver cross-attn O(N T).  Analytical (real architecture constants).
# ==========================================================================
d = 3072; patches = 729; N = 128
tiles = np.arange(1, 11)
T = patches * tiles
full = 4 * (T**2) * d            # QK^T + AV, per attention layer
perc = 4 * N * T * d + 4 * (N**2) * d  # cross-attn + latent self-attn
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.plot(tiles, full/1e9, "-o", color="#e34948", markersize=4, linewidth=1.4,
        label="Full self-attention  $O(T_v^2 d)$")
ax.plot(tiles, perc/1e9, "-s", color="#2a78d6", markersize=4, linewidth=1.4,
        label="Perceiver  $O(N T_v d)$")
ax.axvline(5, color="#888888", linestyle=":", linewidth=0.8)
ax.text(5.1, ax.get_ylim()[1]*0.02, "ours (5 tiles)", fontsize=6.6, color="#555555")
ax.set_yscale("log")
ax.set_xlabel("Number of AnyRes tiles")
ax.set_ylabel("Attention GFLOPs / layer")
ax.grid(True, which="both", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.8, loc="lower right")
save(fig, "flops")
plt.close(fig)
r5 = (4*(729*5)**2*d) / (4*N*(729*5)*d + 4*N*N*d)
print("wrote flops.pdf  FLOP reduction at 5 tiles = %.1fx" % r5)
print("DONE")
