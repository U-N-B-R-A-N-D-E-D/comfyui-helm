# Example outputs

Two takes of the same prompt and seed, differing only in the Helm slider `strength` (the `density` axis,
block 20, on MiniMax H3). No other input changes.

| File | Helm `strength` | Shows |
|---|---|---|
| `hero_strength_0.00.mp4` | 0.00 | baseline (a corridor of glass server racks) |
| `hero_strength_0.25.mp4` | 0.25 | the same corridor, emptier — the racks are gone |

Rendered with the example workflow (`../example_workflows/h3_t2v_helm.json`, the API twin lives in `../api/`), MiniMax H3 (fast preset,
480×864 source, 10 s). That graph chains two sliders (`mood` → `density`); both apply, so each axis moves the
output independently. The `strength` band and axis scale are model-specific — see `../calibration.json`.
Higher `strength` eventually washes the frame; see the "Limitations" section of the top-level README.
