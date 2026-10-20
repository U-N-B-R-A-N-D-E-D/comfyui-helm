# Evolution window — 2026-10-19

Per-render provenance for the fix-vs-push window (Z-Image, seed 11).

- `renders.jsonl` — one line per job (384): `job_id`, `stage` (`sweep`/`window`/`where`), `axis`,
  `mode`, `strength`, `status`, the output `sha256`, the graph `sha256`, and the sidecar path.
  381 rendered; the 3 `where` demos were refused (`mask covers 4096, block has 4128`).
- `renders.summary.json` — counts, the pre-registered criterion, hashes of the batch manifest and the
  labels, and the pack/HAL commits.

The PNGs, the per-render `<name>.png.json` sidecars and the 384 API graphs live outside git at
`~/HAL-archive/measurements/evolution-window/20261019/` (`images/evo/`, `graphs/`, `failures/`).
The verdicts and per-strength metrics are in [`../axis-labels.json`](../axis-labels.json).

Regenerate the index from the archived batch (in `~/HAL/integrations/comfyui-steer-h3`):

```sh
python3 video/write_render_sidecars.py
```
