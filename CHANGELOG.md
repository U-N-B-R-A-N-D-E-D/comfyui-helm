# Changelog

## 0.3.9 — 2026-10-19

- `HelmApply` gains optional `mask_offset` for packed sequences (Z-Image: 32
  text/cond rows before 4096 spatial tokens; H3: rows before the video slice).
- Per-render provenance index for the 2026-10-19 window
  (`docs/data/evolution-20261019/`): 385 jobs, output and graph hashes.
- Disclosed the post-hoc quality ceiling: **10/53** primary push holds,
  **7/53** with saturation <= 0.5 and edge <= 2x baseline (`strict_push`).
- Z-Image room-density axis: `safe_max` corrected to **0.25** (wash at 0.5);
  the example workflow uses 0.25.
- `grid` tooltip matches the code (explicit, never inferred); public docs no
  longer reference private paths.
- Masks documented as injection-only: the delta is zero outside the mask, but
  attention propagates the visible change.

## 0.3.8 — 2026-10-19

- Restored the shipped mood axis, guarded `example_axes/`, truthful README,
  measured prompt-mix benchmark, Z-Image axis, 0.3.8 icon.
