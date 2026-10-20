# Data files

- `zimage-78-axes.json` — the 78 Z-Image directions measured on 2026-10-18 (`axis`, `norm`, `cos`,
  `dosed`). `cos` is the cosine alignment with the mean activation, i.e. the feature the refuted
  predictor used — it is **not** a separation score.
- `zimage-dose-response.json` — 163 measured dose points covering **55 of the 78** axes. The other
  23 axes were captured but not dosed, so no per-axis "holds / washes" verdict exists for them.
- `bench-vs-prompt-mix.json` / `bench2-vs-prompt-mix.json` — the first (invalid: opposite directions)
  and the fair (both start from the same prompt) prompt-mix comparisons. Use the `bench2` one.

**Label history (stated, not hidden):** the 2026-10-18 sweep only reported "roughly 1 in 3 holds"
without a machine-readable per-axis label. The 2026-10-19 run re-derived the labels from those same
renders under a pre-registered criterion and recorded them in `axis-labels.json`. Do not infer labels
from `cos`: the hypothesis that cosine predicts survival is exactly what the 78-axis test refuted.

## Reproducibility

- `bench2-graphs/` — the eight API graphs behind the fair prompt-mix benchmark
  (baseline seed 11/23, Helm 0.5/1.0, mix 0.9/0.8/0.7/0.5/0.3). Submit them with the
  pod-manager (`POST /generate-async`) against a Z-Image-Turbo install; the axis is
  `example_axes/z_room_density.safetensors`.

- `axis-labels.json` — the pre-registered holds/washes labels for the 55 dosed axes under
  `push`/`norm`/`clamp` (2026-10-19), with the per-strength CLIP/sat/edge/lum numbers.

- `evolution-20261019/` — per-render provenance for that window: `renders.jsonl` (385 jobs: the
  384-job window + one unsteered baseline) with each output's `sha256` and graph `sha256`, plus
  `renders.summary.json`.
  The images and per-render sidecars live in
  `~/HAL-archive/measurements/evolution-window/20261019/` (outside git).
