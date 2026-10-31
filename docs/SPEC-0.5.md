# Helm 0.5 spec (core)

Status: implemented in `feat/helm-0.5-core`, unit-tested on synthetic tensors, **not yet run on a GPU**. Nothing in
this document is a measured result.

## The idea

Four small nodes, one data type. Each node has one job and the minimum path is two nodes.

```
Helm · Axis  ->  Helm · Shape  ->  Helm · Mix  ->  Helm · Apply  ->  sampler
 (WHAT)          (HOW MUCH,        (COMBINE)       (WHERE)
                  WHEN)
```

`Helm · Dial` stays as the one-node shortcut (library idea + 0..1 amount, no chaining).

## The nodes

| Node | Class | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| Helm · Axis | `HelmAxis` | `source` (library / prompts), `idea`; optional `helm`, `model`, `clip`, `latent`, `pairs`, `name`, `debias`, `debias_strength`, `steps`, `seed`, `cfg`, `block` | `HELM` |
| Helm · Shape | `HelmShape` | `helm`, `amount` (-1..1); optional `stage`, `split`, `curve`, `end_amount`, `easing`, `keyframes` | `HELM` |
| Helm · Mix | `HelmMix` | `helm_1`, `orthogonalize`; optional `weight_1`, `helm_2..4`, `weight_2..4`, `axis_weights` | `HELM`, `STRING` (report) |
| Helm · Apply | `HelmApplyNode` | `model`, `helm`; optional `cond_only`, `mode` | `MODEL` |

### Axis (WHAT)

- **Library:** the same menu as the Dial (Essentials, Subtle, Custom). An idea made for another model is refused
  with a plain message when a model is connected, and again by Apply.
- **Prompts:** several `with the idea || without the idea` lines, ideally with different subjects. For each pair the
  node records the block's mean activation for both sides; the axis is the **mean of the per-pair differences**. What
  differs from pair to pair (the subject) points in a different direction each time and shrinks; what the pairs
  share stays. It also reports how well the pairs agree (mean cosine with the average) and prints a hint when they
  barely do.
- **Debias:** a fixed list of built-in *nuisance pairs* (`NUISANCE_PAIRS` in `nodes.py`: same scene in other words,
  same template with another subject, a warm/cool and a vivid/muted look) is recorded the same way. Their normalized
  differences span a small subspace; the axis is projected off it (`v - s * Q^T Q v`, `s` = `debias_strength`). The
  reasoning: any prompt change drags a shared "look and colour" component with it, which is what made some of our
  axes read as pink, blue or yellow tints. The nuisance recordings do not depend on the idea, so they are cached for
  the session. If more than 90% of an axis lies in the nuisance subspace the node says so.
- **Saved:** a prompt-built axis is saved to `user/steer/` like Make Axis does, with an *estimated* `safe_max` (0.5%
  of the typical activation norm), never presented as verified.
- **Chaining:** the optional `helm` input is the rgthree "Power Lora Loader" pattern. The new idea is appended to the
  chain; the input object is never modified.

Independent implementation of the published ideas: Activation Addition (Turner et al., arXiv:2308.10248) for adding
a difference of activations, Concept Sliders (arXiv:2311.12092) for a signed continuous control of one concept. No code
was taken from other packs.

### Shape (HOW MUCH and WHEN)

- **Signed amount** -1..+1 maps to `-safe_max .. +safe_max` of each axis (a unit of amount is the axis's own safe range).
- **Time curve (video):** `constant`, `ramp` (from `amount` on the first frame to `end_amount` on the last, with
  `linear`, `ease_in`, `ease_out` or `ease_in_out`), or `keyframes` (`position:amount`, 2 to 8, increasing positions).
  Frames are latent frames, positions are relative (0 first, 1 last).
- **Stage:** `all steps`, `early steps (composition)` or `late steps (finish)`, split at `split` (default 0.5). Progress
  is computed from the sigma ComfyUI passes to the model, normalized against `transformer_options['sample_sigmas']`,
  so a shifted schedule is respected; it falls back to the step counter when the schedule is missing or the sigma lies
  outside it.
- Shape sets amount, curve and stage on **every** idea already in the chain. To shape ideas differently: shape one,
  then chain the next (`Axis -> Shape -> Axis`), or shape separately and combine with Mix.

### Mix (COMBINE)

- Up to four `HELM` inputs, each with a 0-1 weight, plus an optional `axis_weights` list (one per idea, in arrival order).
  Weights multiply; a weight stays within 0-1, so no idea can exceed its safe range.
- **Orthogonalize** (default on): modified Gram-Schmidt, in order, over the **active** ideas (weight and amount not
  zero) that share model, block and width. The first idea is never changed; each later one loses its components along
  the earlier ones and is rescaled to its original length so its amount keeps its meaning. An idea that is almost
  inside the span of the earlier ones is reported and applied unchanged. Inactive ideas never influence the others.
- **Warnings:** different models, or different blocks (those cannot be made independent of each other). They go to the
  `report` output, to the log, and travel inside the `HELM` so Apply prints them too.

### Apply (WHERE)

`MODEL + HELM -> MODEL`, the only 0.5 node that patches the model. It builds one hook per active idea through the same
`_add_axis_hook` and wrapper as `Helm Lab · Apply axis` and the Dial (so the chained hooks, the Aimdo workaround and the
tuple guard are shared). If nothing is active (every amount or weight is 0) it returns the model untouched.

## The `HELM` type (`helm_core.py`)

An immutable `Helm` holds 1 to 16 `HelmAxis` entries and a tuple of warnings. Each `HelmAxis`:

| Field | Meaning |
| :--- | :--- |
| `name`, `label` | file name and human name |
| `vec` | float32 1-D tensor, finite, non-zero |
| `model`, `module`, `block` | where it was made; checked against the connected model |
| `safe_max` | range 0..safe_max the amount maps onto (> 0) |
| `source` | `library`, `pairs` or `other` |
| `amount` | signed, -1..1 (the amount at the start when there is a curve) |
| `weight` | 0..1, set by Mix |
| `curve` | none, or `HelmCurve(kind constant/ramp/keys, end, easing, keys)` |
| `stage`, `split` | `all` / `early` / `late`, and the boundary |
| `info` | up to 16 small text/number/boolean facts (pairs used, debiased, `safe_max_source`, ...) |

Validation is strict at construction and on `from_dict`: types, ranges, NaN/inf, unknown keys, bad curves. `to_dict()` is
plain JSON (`from_dict(to_dict(x))` round-trips); there is no pickle. Nodes never mutate a `HELM`; they return a new one
(this matters for ComfyUI's caching).

## Defaults

- `Axis -> Apply` with no Shape: amount **0.5**, the whole clip, all steps, weight 1.
- `Apply` without Mix applies the chained list as it is.
- The minimum path is two nodes.

## The H3 per-frame token layout

Read from the local ComfyUI core 0.43.2 (`comfy/ldm/minimax/model.py` unless noted):

- The packed sequence is `[text | keyframe cond | refs | audio | video]`; the target video is **always the last segment**
  (comment at line 443, `n_video = latent_t * frame_rows` at 450, appended at 451).
- Rows per frame are `(H/2) * (W/2)` of the padded latent frame (`_frame_grid` 88-92; `_axis_from_sqrt_area` 70-74,
  `n = dim // patch`; padding to the 1x2x2 patch at 652).
- Rows are **frame-major**, then h, then w (`patchify_video` 42-49, `_video_grid` 126-130).
- A block's output `h` is a 2-D `[sequence, hidden]` tensor (592), batch 1 (653-654).
- The sampler hands the wrapper the packed latent shapes: `latent_shapes[0]` is the video latent `[B, C, T, H, W]`
  (`model_base.py` 2228-2230 sets it as a cond; `comfy/samplers.py` 333 passes `c` to `model_function_wrapper`;
  1283-1286 builds the shapes). `model_base.py` 252 copies `transformer_options` before the model runs, which is why
  the layout the model stores in its own copy (`model.py` 669) is **not** visible to Helm and the video span is derived
  from the shapes instead.

So the video rows of a block output are `[S - T*rows_per_frame, S)` and frame `f` owns the `rows_per_frame` rows starting
at `S - T*rows_per_frame + f*rows_per_frame`. `helm_core.h3_video_layout` computes this and **refuses** a shape that cannot fit
the sequence. With a non-constant curve, rows outside the video span (text, references, audio) get amount 0 (a design
choice, to be measured), and a model that is not MiniMax H3 uses the curve's average amount on every token and says so.

I am confident in the mapping from the code, but it has not been checked against a real H3 forward pass, so the curve
ships as **experimental**.

## What is experimental

- **Time curve:** the layout is derived from source and unit-tested on synthetic tensors, never run on real H3
  activations. Whether per-frame steering of the residual stream gives a per-frame visual effect is unknown (attention
  mixes frames in later blocks). Only MiniMax H3 is supported.
- **Stage:** the gating maths and hook path are tested; the visual claim "early = composition, late = finish" is
  untested for these axes.
- **Debias:** the projection maths is tested on synthetic vectors; whether it removes visible tints without weakening the
  idea is not measured. The nuisance list is a first guess; its look pairs could also remove a legitimate colour idea.
- **Mix orthogonalization:** the maths is exact on the vectors; independence of the rendered effect is not measured.
- **`safe_max` of a prompt-built axis:** an estimate (0.5% of the activation norm), calibrated on one Z-Image axis.

## What the GPU benchmark must measure later

1. **Multipurpose generalisation across scenes:** one multi-pair axis (3, 6, 12 pairs) vs a single-pair axis, applied to
   scenes unrelated to the pairs; effect size and subject preservation.
2. **Tint removal:** colour statistics (mean hue/saturation shift) with and without debias, per axis, at equal visible
   effect; and whether debias weakens the intended idea.
3. **Signed symmetry:** amount +a vs -a around the unsteered render; is the effect opposite and of similar size, and does
   the safe range hold on the negative side.
4. **Time curve fidelity:** a ramp and a keyframed curve on H3; does the measured effect per frame follow the curve; the
   effect of zeroing the non-video rows vs the average; the latent frame to pixel frame mapping.
5. **Stage separation:** early-only vs late-only vs all; do early steps change layout and late steps change finish; the
   best default split.
6. **Orthogonality when mixing:** two ideas with and without Gram-Schmidt; cross-talk (moving A changes how B measures);
   the loss of strength from orthogonalizing, and whether the order matters.
