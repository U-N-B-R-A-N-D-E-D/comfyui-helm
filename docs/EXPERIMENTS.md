# Experiments

What we actually measured, and what we did not. Numbers here come from real runs on a
MiniMax H3 pod; the caveats are as important as the results.

## Setup

- **Model:** MiniMax H3 (video), block 20, hidden dim 5376, `fast` preset, 704×1280, 10.1 s.
- **Axis:** `density` — diff-in-means over two contrast prompts (*empty hall* vs *packed hall*),
  shipped as `example_axes/density.safetensors`.
- **Metric, per output:** the mid frame is scored by CLIP (`openai/clip-vit-base-patch32`) as
  `P("an empty room")` vs `P("a room packed with server racks and machinery")` → a single
  `CLIP empty` number. Plus plain image statistics — mean luminance, mean HSV saturation, and
  mean gradient magnitude ("edge density") — and mean-abs-difference vs the baseline frame.

![dose-response](images/dose_response.png)

## A. Dose-response (density axis)

Seed 11, block 20, base prompt; strength swept:

| strength | CLIP empty | saturation | edge density | diff vs base |
|---------:|-----------:|-----------:|-------------:|-------------:|
| 0.00 | 0.067 | 0.176 | 0.007 | — |
| 0.10 | 0.744 | 0.144 | 0.005 | 0.063 |
| 0.20 | 0.644 | 0.105 | 0.004 | 0.100 |
| 0.30 | 0.980 | 0.073 | 0.003 | 0.126 |
| 0.45 | 0.995 | 0.034 | 0.002 | 0.154 |

Replication — **seed 23 at 0.30 = 0.990** (vs 0.980 for seed 11): the effect reproduces.

**Reading.** Raising `strength` moves the frame toward "empty": the CLIP score climbs from
0.07 to ~1.0, saturation and edge density fall monotonically (fewer, flatter surfaces). The
trend is the result; individual points are noisy — note CLIP dips at 0.20 (0.644) after 0.10
(0.744).

**Honest caveat.** Past ~0.45 the frame begins to wash out (see [Limitations](../README.md#limitations)),
and at that point the "empty" reading conflates a real content change with the wash. The CLIP
probe is informative in the low-strength band and uninformative at the collapse.

## B. Is it a direction, or just magnitude? (hue control)

A second axis built from two prompts that differ **only in hue** (cool blue vs warm red),
same composition. If steering were just "more energy" (a magnitude effect), any axis would
look the same. It doesn't:

![axis vs magnitude](images/axis_vs_magnitude.png)

| axis | strength | CLIP empty | saturation | edge density |
|------|---------:|-----------:|-----------:|-------------:|
| density | 0.30 | 0.980 | **0.073** | 0.003 |
| hue | 0.30 | 0.990 | **0.917** | 0.009 |

At the same strength, density **drains** saturation (0.073) while hue **inflates** it (0.917),
and hue **preserves** edges (0.009 vs 0.003). The two directions push different, specific
properties — so the effect is directional, not a generic wash.

**Honest caveat.** The CLIP "empty" score also rises for the hue axis (0.990) even though the
content is preserved. That means the CLIP anchor is confounded at high strength (it tracks the
overall tonal shift, not just emptiness). We report it rather than hide it: **saturation and
edge density are the trustworthy signals here; CLIP is a useful low-strength probe, not a clean
high-strength one.**

## C. The same contrast in a second model (partial)

**Goal.** Capture the *same* contrast in a second model and ask whether the direction does the same
*kind* of thing — the first step toward portability.

**Done.** Z-Image (image, 30 layers, dim 3840) captured on the pod through the **job path**, layer 15,
identical *empty vs packed hall* prompts:

| Z-Image `density` | CLIP empty | luminance | saturation | edge density |
|--:|--:|--:|--:|--:|
| 0.00 | 0.000 | 0.496 | 0.065 | **0.018** |
| 1.50 | 0.011 | 0.891 | 0.005 | 0.004 |
| 3.00 | 0.951 | 0.947 | 0.012 | **0.001** |

![Z-Image density](images/zimage_density.png)

**Reading.** Edge density — the content signal — falls monotonically in **both** models (H3: `0.007 → 0.002`; Z-Image: `0.018 → 0.001`). A "density" direction is not unique to H3's architecture.

**What this is NOT.** It is **not vector transfer**. A dim-5376 axis and a dim-3840 axis are different
objects; we compared the *effect of the same contrast in each model*, not a carried vector. Carrying
one model's axis into another's space is untested and needs an alignment.

**Caveat.** At Z-Image `s=3` the CLIP "empty" score (0.951) coincides with a near-white wash
(luminance 0.947), so that point is the wash, not a clean semantic reading — the same CLIP confound
noted in §B.

**Infra note.** A first attempt over SSH was killed by the pod's **idle watchdog** (SSH is not
counted as activity); the successful run went through a ComfyUI API job.

## Knobs not yet measured

Three controls are implemented and off — in their default state they reproduce every number above
exactly. None of them has been through a dose-response re-measure on the pod yet, so treat them as
untested knobs, not results:

- **`cond_only`** (`HelmCollect`, `HelmApply`; default `False`). Restricts collection and injection to
  the conditional batch items, leaving the unconditional pass untouched. This is the recommended
  setting for CFG models; the shipped axes and every table here were measured with it off.
- **Windowing** (`HelmApply` `start_percent` / `end_percent`; defaults `0.0` / `1.0`). Applies the axis
  only within a fraction of the sampling schedule. The default window spans the whole run, i.e. today's
  behavior.
- **Per-slider strength** (`app/run_helm.py --set axis=strength`). Sets a strength per `HelmApply` node
  instead of one global value. Exercised in unit tests only; not run against a live server.

A re-measure covering these is pending. Until it exists, the published band and figures stand only for
the default (off) settings.

## What this is not

- Not a claim that "density" or "mood" are *concepts*. They are low-level directions; the wash
  at high strength is the signature of pushing a sample off the model's manifold.
- Not a comparison against **LoRA** baselines (not measured). The prompt-only (`ConditioningAverage`) comparison is §E, with the fair design.
- One model (H3) for A and B, one contrast pair each.

## D. The probe: what a per-block profile tells us, and what it does not (Z-Image)

`HelmProbe` exists to answer "which block?" without a manual sweep: it reads *every* block over two
contrastive passes and ranks them by how far apart the two prompts sit, as
`||mu_pos - mu_neg|| / mean(||mu_pos||, ||mu_neg||)` — scale-free, so blocks compare to each other.
Run on Z-Image (30 layers, dim 3840, `NextDiT`, 6-step turbo graph, the *same* empty-vs-packed contrast
as §C):

| block | separation | cosine | \|mu\| | \|dmu\| |
|--:|--:|--:|--:|--:|
| 0 | 0.213 | 0.978 | 27 | 5.8 |
| 3 | 0.082 | 0.997 | 79 | 6.5 |
| 7 | 0.034 | 0.9995 | 230 | 7.8 |
| 9 | 0.033 | 0.9997 | 334 | 11.1 |
| 15 | 0.012 | 0.9999 | 964 | 11.5 |
| 18 | 0.011 | 0.9999 | 1212 | 13.5 |
| 29 | 0.085 | 0.996 | 1112 | 94 |

**Data in the repo:** the per-axis table (78 rows: `axis`, `norm`, `cos`) is [`data/zimage-78-axes.json`](data/zimage-78-axes.json); the 163 measured dose points behind the bands are [`data/zimage-dose-response.json`](data/zimage-dose-response.json).

**Reading — this is data.** The separation is structural, not noise: it is largest at the first blocks,
falls monotonically into the middle (minimum at block 18), and rises again at the end. The two prompts
pull hardest at the extremes of the stack.

**What it does NOT do: pick the block.** The band *verified by rendering* is layers 9–21 (default 15,
`calibration.json`). The probe's highest-separation block inside the middle 50% is **7** — two blocks
below the band — and the naive global peak is **block 0**, which we already know destroys the frame.
Trust the global peak and you steer exactly where any perturbation wrecks the image.

**A strength estimator we built, measured, and removed.** From the H3 numbers (a perturbation ≈30% of the
block's activation norm across the verified band) we derived `strength ≈ 0.3 × |mu| / |dmu|`. On Z-Image
it returns **8.8** at the peak, against a verified band of **0.5–3** — an order of magnitude out. So the
estimator is not in the code: a number that wrong, under the word "suggested", is worse than no number.
The usable scale is model-specific; confirm it with the 1–2 renders `calibration.json` already asks for.

**Conclusion.** `HelmProbeReport` returns the separation profile and the middle-band peak *as a
measurement* — a starting point for a short sweep, not an auto-calibrator. **Caveat:** one model, one
contrast, one graph. H3 was not probed: the GPU window was shared with another workload, and the Z-Image
result already answers the question this feature was built on.

## E. Helm vs prompt mixing (`ConditioningAverage`) — fair re-test, 2026-10-19

The first version of this test was **wrong**: the baseline used the *empty* prompt, Helm pushed further
toward empty, while `ConditioningAverage` at 0.5 moved toward *packed*. It compared opposite directions.

This is the fair design: **both methods start from the same full-room prompt** and are asked to move
toward empty. Z-Image-Turbo, block 15, `cfg 1.0`, 8 steps, `res_multistep`. Axis captured in-session
(`bench`, pos = empty, neg = packed). Seeds 11 and 23.

Prompts: full = `the same room packed with server racks, cables and crates, documentary photography`;
empty = `an empty room with a window and a bare floor, documentary photography`.
Mix: `ConditioningAverage(conditioning_to = full, conditioning_from = empty, strength = s)`; lower `s`
moves further toward empty.

| variant | lum | sat | edge | P(empty) | P(packed) |
|:--|--:|--:|--:|--:|--:|
| full room, seed 11 (baseline) | 0.317 | 0.239 | 0.0266 | 0.000 | 1.000 |
| full room, seed 23 | 0.307 | 0.190 | 0.0262 | 0.000 | 1.000 |
| **Helm 0.5, seed 11** | **0.472** | 0.156 | **0.0209** | 0.000 | 1.000 |
| Helm 0.5, seed 23 | 0.473 | 0.186 | 0.0162 | 0.000 | 1.000 |
| Helm 1.0, seed 11 | 0.640 | 0.241 | 0.0081 | 0.000 | 1.000 |
| mix 0.9 | 0.312 | 0.220 | 0.0265 | 0.000 | 1.000 |
| mix 0.8 | 0.314 | 0.205 | 0.0269 | 0.000 | 1.000 |
| mix 0.7 | 0.309 | 0.183 | 0.0258 | 0.000 | 1.000 |
| mix 0.5 | 0.344 | 0.088 | 0.0133 | 0.000 | 1.000 |
| mix 0.3 | 0.413 | 0.045 | 0.0098 | **0.998** | 0.002 |

CLIP anchors: `an empty room` / `a room packed with server racks and machinery`. Edge = mean gradient
magnitude. Raw numbers: [`data/bench2-vs-prompt-mix.json`](data/bench2-vs-prompt-mix.json).
Images: [`images/bench2/strip.png`](images/bench2/strip.png).

**Reading.**

- **Helm is a graded control at the same framing.** At 0.5 the room still reads as the packed room
  (P(packed) = 1.0) but the frame brightens (lum +0.16) and loses detail (edge -21%), reproducibly across
  seeds 11 and 23. At 1.0 it is visibly washed (edge -70%) — the documented failure mode.
- **The prompt mix is a step, not a slider, on this model.** Between 0.9 and 0.7 it is visually a no-op
  (edge within ±1%); it only moves hard at 0.5 (-50%) and flips the scene at 0.3 (P(empty) = 0.998).
- **They are different controls.** The mix replaces the scene; Helm grades the look at the same prompt
  and framing. That is the defensible claim — not "the mix does nothing" (it does, at 0.3).

**Caveats.** One model (Z-Image-Turbo), one contrast pair, two seeds; at 0.5 part of Helm's brightening
may be the wash the docs describe. A LoRA baseline is still not measured.


## F. Fix vs push, and the step window — measured 2026-10-19

Pre-registered before the run (no tuning after): an axis **holds** at a mode if (a) its
CLIP(pos)-CLIP(neg) score moves monotonically across the three strengths in one direction, (b) edge
density stays >= 50% of the unsteered baseline, and (c) |luminance - baseline| <= 0.20 at every point.
Otherwise it **washes**.

Design: 55 dosed Z-Image axes, the **same** strengths the 2026-10-18 dose sweep used for `push`
(reused renders, not regenerated), with `norm` (rescale each token to its original norm) and `clamp`
(move the projection inside a `clamp_k`*||x|| budget) re-rendered at those strengths; 55 step-window
renders (`start=0.6`, same strength). Seed 11, block from each axis header.

| mode | holds | washes | unmeasured |
|:--|--:|--:|--:|
| push (published) | **10** | 43 | 2 |
| norm | **1** | 52 | 2 |
| clamp | **2** | 51 | 2 |

**Reading — the "fix instead of push" hypothesis is refuted under this criterion.** Norm-preserving and
projection-clamping do **not** raise the survivor rate; they lower it. The mechanism is visible in the
per-axis data: `norm`/`clamp` collapse luminance (e.g. `z_aisle_width` from 0.453/0.248 down to 0.072 at
the top strength; `z_shadow_hardness` to 0.003) while `push` keeps the frame lit. With an axis that is
large relative to the activation, preserving the norm after the addition does not remove the wash — it
keeps the frame in the dark. One axis (`z_pro_mist_diffusion`) is the exception: its baseline contrast
is inverted (CLIP = -0.29), and clamp is the only mode that resolves it.

**Post-hoc quality ceiling (not part of the pre-registered criterion).** The primary labels have
no upper bound on saturation or edge density, so a blown-out frame can still pass. Applying an
explicit ceiling (saturation <= 0.5 and edge density <= 2x the unsteered baseline) to the same three
strengths leaves **7/53** push holds: `z_aisle_width`, `z_corridor_orientation`, `z_motion_blur`,
`z_overcast_vs_direct`, `z_raking_light_angle`, `z_surface_reflection_intensity`,
`z_volumetric_beams`. The three primary survivors it excludes are
`z_saturated_vs_muted_stock`, `z_specular_reflection_density` and
`z_specular_reflection_intensity` (their top dose blows the edge/sat ceiling). The pre-registered
labels are unchanged; `strict_push` in `data/axis-labels.json` records this secondary check. The
specular-density axis is clean at 1.5 (edge 1.6x baseline); its 4.44 top dose is not, so use the
safe band.

**Step window.** Applying `push` only in the last 40% of the schedule moves the frame less: on
`z_aisle_width` at the same strength, luminance 0.404 -> 0.275 and edge 0.0227 -> 0.0297 (structure is
better preserved), but it does not rescue a bad direction (`z_shadow_hardness` still washes). So "when"
is a real, cheap lever — not a fix for a broken axis.

**Where (regional masks).** Z-Image packs 32 text/cond rows before its 4096 spatial tokens, so a
64x64 mask needs `mask_offset=32` (4128 = 32 + 4096); without it the node refuses. Three 1024^2 demos
(left/floor/center, norm 1.0, seed 11) render with the offset. Against an unsteered baseline at the
same size, prompt and seed, the change is stronger inside the mask (mean |delta| 0.17-0.25 vs
0.10-0.18 outside; ratio 1.4-1.7) but **not confined**: >98% of outside pixels also move. The mask
controls where the delta is injected; attention spreads the visible change. Raw: `where.json` in the
archive (the companion media archive (outside git; hashes in `data/evolution-20261019/renders.jsonl`)).

Raw data: [`data/axis-labels.json`](data/axis-labels.json). Images: the per-axis renders live in
the companion media archive (outside git; hashes in `data/evolution-20261019/renders.jsonl`).
