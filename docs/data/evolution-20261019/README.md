# Evolution window — 2026-10-19

Per-render provenance for the fix-vs-push window (Z-Image, seed 11).

- `renders.jsonl` — one line per job (385): `job_id`, `stage` (`sweep`/`window`/`where`/`base`),
  `axis`, `mode`, `strength`, `status`, the output `sha256`, the graph `sha256`, and the sidecar path.
  All 385 rendered: 384 window jobs + 1 unsteered baseline used to measure the `where` demos. The
  three `where` jobs pass `mask_offset=32` (Z-Image: 32 text/cond rows + 4096 spatial = 4128).
- `renders.summary.json` — counts, the pre-registered criterion, hashes of the batch manifest and the
  labels, and the pack/HAL commits.

The PNGs, the per-render `<name>.png.json` sidecars and the 384 API graphs live in the
companion media archive (outside git): `images/evo/`, `graphs/`, `failures/`.
The verdicts and per-strength metrics are in [`../axis-labels.json`](../axis-labels.json).

Regenerate the index from the archived batch with the sidecar writer (schema `helm.render/1`):

```sh
python3 video/write_render_sidecars.py
```
