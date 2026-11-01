/* Every number on the slides comes from here, and every number here comes from
   the camera-ready paper (paper-ica26/main.tex, identical to camera-ready/src)
   or from results/eval_results.json. The source is noted beside each block. */

window.AP = {
  // main.tex, Table 2 (tab:main)
  models: [
    { key: "llava", name: "LLaVA-NeXT-7B", params: "7B", json: 99.9, schema: 99.9, f1: 0.134, fm: 0.001, mae: 0.344, r: 0.226, bert: 0.847, comp: 0.590 },
    { key: "intern", name: "InternVL2-8B", params: "8B", json: 100.0, schema: 100.0, f1: 0.336, fm: 0.005, mae: 0.385, r: 0.624, bert: 0.848, comp: 0.611 },
    { key: "qwen", name: "Qwen2-VL-7B", params: "7B", json: 99.9, schema: 99.9, f1: 0.287, fm: 0.001, mae: 0.224, r: 0.666, bert: 0.830, comp: 0.615 },
    { key: "ours", name: "AgriPerceiver", params: "4.6B", json: 99.7, schema: 99.7, f1: 0.645, fm: 0.486, mae: 0.067, r: 0.855, bert: 0.921, comp: 0.810 },
  ],

  // Fig. 2 data. Baselines: scripts/make_figs.py (from paper/RESULTS.md §2);
  // ours: eval_results.json classification.type.per_class. Support from Table 3.
  classes: ["Fungal", "Unknown", "Bacterial", "Deficiency", "Pest", "Viral"],
  support: { Fungal: 1521, Unknown: 1535, Bacterial: 692, Deficiency: 618, Pest: 351, Viral: 345 },
  perClassF1: {
    llava:  { Fungal: 0.301, Unknown: 0.493, Bacterial: 0.000, Deficiency: 0.006, Pest: 0.000, Viral: 0.006 },
    intern: { Fungal: 0.599, Unknown: 0.689, Bacterial: 0.003, Deficiency: 0.475, Pest: 0.227, Viral: 0.022 },
    qwen:   { Fungal: 0.637, Unknown: 0.634, Bacterial: 0.025, Deficiency: 0.386, Pest: 0.017, Viral: 0.022 },
    ours:   { Fungal: 0.813, Unknown: 0.827, Bacterial: 0.560, Deficiency: 0.605, Pest: 0.541, Viral: 0.525 },
  },

  // eval_results.json classification.type.confusion_matrix
  // rows = truth, cols = prediction, order: bacterial, deficiency, fungal, pest, unknown, viral
  cmLabels: ["Bacterial", "Deficiency", "Fungal", "Pest", "Unknown", "Viral"],
  cmShort: ["Bac", "Def", "Fun", "Pst", "Unk", "Vir"],
  confusion: [
    [360, 79, 145, 56, 15, 37],
    [22, 437, 25, 16, 62, 56],
    [121, 32, 1274, 52, 18, 24],
    [32, 9, 80, 203, 20, 7],
    [30, 169, 61, 68, 1165, 42],
    [28, 101, 27, 4, 3, 182],
  ],

  // eval_results.json calibration.bins (only non-empty bins exist)
  calibration: {
    ece: 0.139,
    bins: [
      { lo: 0.4, hi: 0.5, n: 17, conf: 0.5, acc: 0.118 },
      { lo: 0.6, hi: 0.7, n: 792, conf: 0.7, acc: 0.552 },
      { lo: 0.7, hi: 0.8, n: 1354, conf: 0.797, acc: 0.571 },
      { lo: 0.8, hi: 0.9, n: 1405, conf: 0.866, acc: 0.744 },
      { lo: 0.9, hi: 1.0, n: 1494, conf: 0.982, acc: 0.912 },
    ],
  },

  // main.tex Sect. 4.6, Eqs. (9)-(10): T_v = 729 per tile, N = 128, d = 3072
  complexity: { perTile: 729, N: 128, d: 3072, deployedTiles: 5 },

  // main.tex Sect. 5 (data pipeline)
  data: { images: 117635, kept: 101301, train: 96239, test: 5062, tau: 0.5,
          buckets: [ { name: "easy", range: "[0.8, 1.0]", w: 1.0, share: 0.56 },
                     { name: "medium", range: "[0.6, 0.8)", w: 0.8, share: 0.35 },
                     { name: "hard", range: "[0.4, 0.6)", w: 0.5, share: 0.09 } ] },

  // main.tex Table 5 (tab:params), in millions
  params: { siglip: 400, bridge: 390.98, lora: 35, phi: 3820, total: 4650 },
};
