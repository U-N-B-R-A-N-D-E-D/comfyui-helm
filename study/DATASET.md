# Data card — Helm Study 01

This is the derived data behind [`README.md`](README.md). The source of truth is [`../docs/data/`](../docs/data) and [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md); the CSVs here are normalised copies with a stable schema. Rebuild with `make -C study`.

## Scope

- **Model:** Z-Image (Lumina2), 30 layers, hidden dim 3840, block recorded per axis.
- **Sampling:** seed 11, three strengths per axis in `push`, `norm` and `clamp`; a second seed (23) only for the prompt-mix benchmark and the H3 dose check.
- **Window:** the 2026-10-19 capture. 78 directions captured; 55 dosed; 53 labelled; 2 unmeasured.
- **Unit:** one row per direction, one row per dose point, one row per render job.
- **Not included:** the rendered images (outside git), the H3 video renders, and the KCA client production renders. The 452 `kca-z-image-turbo` renders are a client's work and are never part of this study.

## Files

| file | rows | grain | source |
|---|---:|---|---|
| `data/axes.csv` | 78 | direction | `docs/data/zimage-78-axes.json` + `axis-labels.json` |
| `data/labels_long.csv` | 544 | direction × mode × strength | `docs/data/axis-labels.json` |
| `data/doses.csv` | 163 | direction × dose | `docs/data/zimage-dose-response.json` |
| `data/window.csv` | 55 | direction × window | `docs/data/axis-labels.json` |
| `data/bench_prompt_mix.csv` | 10 | variant | `docs/data/bench2-vs-prompt-mix.json` |
| `data/provenance.csv` | 385 | render job | `docs/data/evolution-20261019/renders.jsonl` |
| `data/provenance_summary.csv` | 5 | stage × mode | derived from `provenance.csv` |
| `results/metrics.csv` | 78 | direction | derived from `axes.csv` |
| `results/survival_by_category.csv` | 5 | family | derived |
| `results/model_matrix.csv` | 7 | model | `calibration.json` + README |
| `results/content_calendar.csv` | 58 | post | derived |

## Schemas

### `data/axes.csv`

| column | meaning |
|---|---|
| `axis` | direction id (`z_...`) |
| `category` | curated family: light, camera, material, atmosphere, geometry |
| `cos` | cosine alignment with the mean activation (the refuted screen) |
| `norm` | axis norm as captured |
| `dosed` | `yes` if the direction was measured at three strengths |
| `label_push` | `holds`, `washes` or `unmeasured` under the pre-registered criterion |
| `strict_push` | post-hoc ceiling label; `holds`, `washes`, or empty for non-survivors |
| `base_*` | unsteered baseline: CLIP contrast, edge, luminance, saturation |
| `top_strength` | strongest push dose measured |
| `clip_delta` | CLIP contrast at the top push dose minus baseline |
| `edge_ratio_top` | edge density at the top push dose divided by baseline |
| `lum_delta_max` | maximum absolute luminance change across the three doses |
| `sat_top` | saturation at the top push dose |
| `wash_onset` | first dose crossing edge < 0.5×, saturation > 0.5 or \|lum Δ\| > 0.20 |
| `monotone` | whether the CLIP contrast moved in one direction across the three doses |
| `spearman` | Spearman correlation between strength and CLIP contrast |

### `data/labels_long.csv`

One row per baseline, `push`, `norm` and `clamp` point. Columns: `axis`, `category`, `mode`, `strength`, `lum`, `sat`, `edge`, `clip`, `label`. This is the tidy table to use for a new chart.

### `data/doses.csv`

The 163 measured dose points. `edge` is on a different scale from `axes.csv` (the dose sweep recorded gradient magnitude on 0-255); do not mix the two columns in one plot. Use `axes.csv` for the survival labels and `doses.csv` for the raw dose sweep.

### `data/provenance.csv`

One row per render job: `axis`, `stage` (`sweep`, `window`, `where`, `base`), `mode` (`norm`, `clamp`, `push`, `none`), `strength`, `status`, `output`, `sha256`, `graph_sha256`, `job_id`. The original JSONL is the authority; this CSV is a convenience view.

## Provenance

- 385 jobs, all `done`, 0 failed.
- Stages: sweep 326, window 55, where 3, base 1.
- Modes: norm 166, clamp 163, push 55, none 1.
- Pack commit `b555cc5505ff1ce4482f96935463542fe0e4e86c`.
- HAL commit `cb4d7d7c3bbec31f531538b0f21b2e4d1dabf141`.
- Batch manifest and labels hashes are in [`../docs/data/evolution-20261019/renders.summary.json`](../docs/data/evolution-20261019/renders.summary.json).

## Labels

The primary labels are the pre-registered criterion in `axis-labels.json`:

- CLIP contrast moves monotonically across the three strengths in one direction;
- edge density stays at least 50% of the unsteered baseline;
- luminance stays within 0.20 of the baseline at every point.

`strict_push` is a post-hoc ceiling added after the labels were fixed: saturation <= 0.5 and edge <= 2× baseline. It leaves 7 of 53. The primary labels are unchanged.

## What is not measured

- 23 of the 78 directions were captured but not dosed; they have no hold/wash label.
- 2 of the 55 dosed directions have only two points and no label (`matte_vs_glossy`, `refraction_index_shift`).
- Subject identity, camera movement and lighting are not measured; the study uses CLIP contrast, saturation, edge density and luminance as proxies.
- The mask localises the injection, not the visible effect: attention spreads the change outside the mask.
- A LoRA baseline is not measured.

## Licence

- Code and scripts: MIT, same as the pack.
- Labels, metadata and CSVs: CC BY 4.0.
- Renders: not included; the media archive is outside git. If you redistribute a render, check the model's licence first.
- Axis files: MIT, as shipped in the pack.

## Citation

If you use the tables, cite the pack and the study:

```bibtex
@misc{helm2026study,
  title  = {Helm Study 01: 78 directions, 55 dosed},
  author = {{U N B R A N D E D}},
  year   = {2026},
  note   = {Derived from docs/data in comfyui-helm},
  url    = {https://github.com/U-N-B-R-A-N-D-E-D/comfyui-helm/tree/main/study}
}
```
