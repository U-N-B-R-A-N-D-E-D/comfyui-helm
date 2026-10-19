# Data files

- `zimage-78-axes.json` — the 78 Z-Image directions measured on 2026-10-18 (`axis`, `norm`, `cos`,
  `dosed`). `cos` is the cosine alignment with the mean activation, i.e. the feature the refuted
  predictor used — it is **not** a separation score.
- `zimage-dose-response.json` — 163 measured dose points covering **55 of the 78** axes. The other
  23 axes were captured but not dosed, so no per-axis "holds / washes" verdict exists for them.
- `bench-vs-prompt-mix.json` / `bench2-vs-prompt-mix.json` — the first (invalid: opposite directions)
  and the fair (both start from the same prompt) prompt-mix comparisons. Use the `bench2` one.

**Known gap (stated, not hidden):** the earlier sweep summarized the result as "roughly 1 in 3 holds",
but the per-axis survivor label and the exact criterion were not recorded in a machine-readable file.
Do not infer labels from `cos`: the hypothesis that cosine predicts survival is exactly what the
78-axis test refuted.
