<div align="center">

# H E L M (`comfyui-helm`)

**A prompt is a switch, not a dial. Helm gives you the dial.**

In practice: more fog, less clutter, more cinematic light — one slider, no prompt rewrite.

**A ~16–22 KB `.safetensors` axis. No training. MIT.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Format: safetensors](https://img.shields.io/badge/axes-.safetensors-brightgreen.svg)](#how-it-works)
[![Tests: 77 passed](https://img.shields.io/badge/tests-77%20passed-success.svg)](docs/HARDENING.md)
[![Comfy Registry](https://img.shields.io/badge/Comfy%20Registry-0.3.10%20Active-success.svg)](https://registry.comfy.org/nodes/comfyui-helm)

[**See it**](#see-it) · [**Benchmark**](#the-honest-benchmark) · [**Start with Z-Image**](#start-with-z-image) · [**Build an axis**](#build-your-own-axis-in-60-seconds) · [**All workflows**](#all-workflows) · [**Experiments (78 axes)**](docs/EXPERIMENTS.md) · [**Roadmap**](docs/ROADMAP.md)

</div>

![Z-Image interior set: base vs steered](docs/images/zimage_interior_before_after.png)
*Z-Image, same prompt and seed. Only the Helm dial moved.*

Here's the honest version of why this exists. A prompt can't do "a little." Change one word and the whole frame changes; make the prompt longer and it leaks into the scene; mix two prompts together and the room stays packed right up until the moment the scene just *flips* (there's a fair benchmark for that below). What I wanted was the middle ground: take a direction the model already knows, and dial it in with a single number. Same framing, graded, no training.

**Steer what it knows. Train what it doesn't.**

That's Helm — training-free activation steering for ComfyUI and the Comfy Developer Platform. You feed it two contrasting passes and it extracts a latent direction into a tiny `.safetensors` file (the density axis that ships with it is 21,888 B — about 22 KB). Then you steer video and image DiTs with one scalar. Zero training. Zero weight modification. MIT-licensed.

## See it

<p align="center">
  <a href="https://youtu.be/MxisBKxoDSQ" title="Watch the Helm demo on YouTube">
    <img src="docs/images/demo.jpg" alt="Helm demo — same prompt, same seed, one number changed" width="70%"/>
  </a>
</p>
<p align="center">
  <a href="https://youtu.be/MxisBKxoDSQ"><b>▶ Watch the 47s demo on YouTube</b></a> — same prompt, same seed, one number changed.
</p>

Move a single slider during sampling and the scene shifts along an internal direction the model already knows. On the measured H3 density pair, the change lands in the *content* while the room's floor line and ceiling stay put — the caption below is the honest version of that. Identity and lighting preservation are **not** measured yet; that's in [Limitations](#limitations).

| **`strength = 0.00`** (baseline) | **`strength = 0.25`** (`density`, block 20) |
| :---: | :---: |
| ![Baseline 0.00](examples/hero_strength_0.00.png) | ![Steered 0.25](examples/hero_strength_0.25.png) |
| *MiniMax H3 (video, 10 s): a corridor of glass server racks.* | *Same prompt, same seed: the racks leave the room; ceiling and floor geometry hold.* |

### One axis, any scene

An axis isn't tied to the prompt that created it. Capture a mood or density direction once (about 22 KB or less as a `.safetensors` file), then dial it into portraits, landscapes, night exteriors, or architectural shots:

![One axis across portrait, forest, city, and interior](docs/images/one_axis_any_scene.png)
*The same axis applied to four different prompts — not a vector transfer test.*

![Helm nodes chained inside ComfyUI](docs/images/ui_in_comfyui.png)

## The honest benchmark

We tested **78 directions** on Z-Image, dosed **55**, and labelled **53**. Under the pre-registered criterion, **10/53** hold in `push`; a post-hoc wash ceiling leaves **7/53**. **43 wash out.**

That is the point. Helm is a measured control, not a magic prompt. Pick a survivor, stay inside its band, and calibrate the wash point for your model.

The fair baseline is prompt mixing (`ConditioningAverage`), same starting prompt, two seeds. At 0.5 the mix is visually a no-op until it flips the scene; Helm grades the same prompt (luminance 0.317 → 0.472, edge 0.0266 → 0.0209). Different controls, not a slower slider. [`§E`](docs/EXPERIMENTS.md#e-helm-vs-prompt-mixing-conditioningaverage--fair-re-test-2026-10-19).

Data: [`zimage-78-axes.json`](docs/data/zimage-78-axes.json), [`zimage-dose-response.json`](docs/data/zimage-dose-response.json), [`axis-labels.json`](docs/data/axis-labels.json). Negative results: [`§F`](docs/EXPERIMENTS.md#f-fix-vs-push-and-the-step-window--measured-2026-10-19).

## Why Helm?

Higgsfield put cinematic look and set controls behind a closed cloud studio. Helm brings deterministic, training-free look & set steering straight into ComfyUI, on open weights:

- **A small file, not a LoRA or checkpoint.** Show the model two contrasting examples (two prompts or two images). Helm records the difference in the internal transformer activations (`axis = mean_pos - mean_neg`) and saves a self-describing `.safetensors` file.
- **A byte-exact zero point.** Set `strength = 0.0` and Helm contributes nothing to the sampled state. The model weights are never touched.
- **Stackable & clamped (a 0 → 1 dial).** Chain several `HelmApply` nodes (mood → density) on the same model patcher. The built-in `safe_max` clamp and a normalized 0.0–1.0 dial keep you inside the verified band.
- **Engineered for real DiT runtimes.** It automatically bypasses MiniMax H3's aimdo per-block memory compiler during hook execution, enforces visual-stream tuple safety on dual-stream architectures, and runs both interactively in ComfyUI and headlessly via the Comfy SDK.

## Start with Z-Image

H3 is the flagship, but it needs a 21 GB int8 UNet and a 32B text encoder. **Start with Z-Image**, which runs on a consumer GPU.

1. **Install**

   ```bash
   comfy node install comfyui-helm@0.3.10
   ```

   Or clone into `ComfyUI/custom_nodes` and restart ComfyUI.

2. **Open the template**

   Templates → search `helm` → `z_image_helm.json`. It loads `z_room_density.safetensors` (block 15, dim 3840, `safe_max 0.25`).

3. **Move the dial**

   Run at `0.0`, then `0.10`, then `0.25`. Same prompt, same seed. The pack clamps at the recorded `safe_max`; stop at the wash point for your axis.

The H3 templates are the flagship video path. They need the third-party `MiniMaxH3Director` node and do not fit in 24 GB. See [All workflows](#all-workflows).

## Build your own axis in 60 seconds

Two contrast passes → one file:

`HelmCollect(tag=pos)` + `HelmCollect(tag=neg)` → `HelmVector`.

Point `HelmApply` at the result. `block=-1` reads the block from the file header; `safe_max` keeps the dial inside the measured band. No training, no adapter, no checkpoint.

**Download the 3-axis pack:** [`example_axes/`](example_axes/).

- `z_room_density.safetensors` — Z-Image, block 15, dim 3840, `safe_max 0.25`
- `mood.safetensors` — MiniMax H3, block 20, dim 5376
- `density.safetensors` — MiniMax H3, block 20, dim 5376

The H3 axes are ~22 KB; the Z-Image axis is ~16 KB. Point `HelmApply` at any of them.

## All workflows

Prebuilt axes ship inside `example_axes/` and resolve automatically on a fresh clone. The H3 axes (`mood.safetensors`, `density.safetensors`) were captured on **MiniMax H3** (block 20, dim 5376); the Z-Image axis (`z_room_density.safetensors`) is block 15, dim 3840.

Open ComfyUI's **Templates** browser, search **`helm`**, or drag a workflow from `example_workflows/` onto the canvas:

**What each path actually needs — no hidden steps:**

| Path | Nodes | Models | Hardware |
| :--- | :--- | :--- | :--- |
| **Z-Image** (`z_image_helm.json`) | core ComfyUI nodes only | Z-Image / Lumina2 weights | runs on a consumer GPU |
| **MiniMax H3** (`h3_*.json`) | **+ `MiniMaxH3Director` (third-party node pack)** | H3 UNet (~21 GB int8) + Qwen3-VL 32B text encoder | the int8 UNet does **not** fit in 24 GB |

| Workflow | Model | What it does |
| :--- | :--- | :--- |
| **`z_image_helm.json`** | **Z-Image / Lumina2** | Steer image generation across verified layers 9–21. |
| **`h3_t2v_helm.json`** | **MiniMax H3** (Video) | **The flagship app:** type a prompt, move the `mood` & `density` sliders, hit Queue. |
| **`h3_i2v_helm.json`** | **MiniMax H3** (I2V) | Image-to-video generation with live activation steering. |
| **`krea2_helm.json`** | **Krea2** | Image generation with `HelmApply` wired between LoRA loader and sampler. *(Template wired; not yet execution-validated.)* |
| **`krea2_helm_collect.json`** | **Krea2** | Two-pass collection graph to build a custom axis from text contrast. *(Template wired; not yet execution-validated.)* |
| **`krea2_helm_image_axis.json`** | **Krea2** (Img2Img) | Extract an axis directly from two contrast images — no contrast prompt needed. *(Template wired; not yet execution-validated.)* |
| **`helm_vector.json`** | *Model-agnostic* | `HelmVector → PreviewAny`: subtract two collected tags into a reusable `.safetensors` axis. |
| **`ltx_t2v_helm.json`** | **LTX-2.3** (Video) | Official LTX blueprint wired with `HelmApply` (structural hook ready; `tested: false`). |
| **`api/h3_t2v_helm_api.json`** | **Comfy SDK / API** | API-format workflow driven by `app/run_helm.py`. |

Measured with Helm: MiniMax H3 t2v (`mood`, `density`) and Z-Image. The models used for generation elsewhere (Krea 2 stills, Qwen Image Edit, LTX/Flux tests) are **not** Helm-measured: Krea 2 templates are wired without a calibration entry, LTX ships `tested: false`, Qwen-Image/SD3 are refused by the tuple guard until declared/measured, and Flux is declared but has no dose. No Helm claim rests on them.

## How It Works

### The node suite (category: Helm)

- **`HelmCollect` (MODEL → MODEL):** hooks `diffusion_model.<blocks>[K]`, accumulates the mean activation across sampling steps, and writes `ComfyUI/user/steer/<tag>.safetensors` once per run. Includes `cond_only` to isolate conditional batch items.
- **`HelmVector` (tags → STRING):** computes `axis = mean_pos - mean_neg` and saves a self-describing `.safetensors` axis with its dim, block, model, module, norm, and optional `safe_max` embedded in the header.
- **`HelmApply` (MODEL → MODEL):** injects `x + strength * axis` at block K during sampling. `block = -1` adopts the block recorded in the header; the dial (0.0–1.0) scales inside `safe_max`; `start_percent`/`end_percent` window the injection; chained nodes compose. Optional: `mode` — `push` (the published one), `norm` (rescale each token to its original norm), or `clamp` (move the projection inside a `clamp_k·‖x‖` budget); plus `mask` + `layout` (`h_w`, `t_h_w`, `h_w_t`) + `grid` (H,W[,T]) + `mask_offset` (the leading packed rows to skip) for a regional application. `mask_offset` is required when the model packs extra rows before the visual tokens: Z-Image has 32 rows before its 4096 spatial tokens (`grid=64,64`, `mask_offset=32`), and H3 packs `[text|cond|audio|video]`, so the offset is the row count before the video slice. The mask is refused unless `mask_offset + grid` tokens == block sequence. Every run logs the block shape once (`[helm] shape debug: ...`) so the token layout is discovered, never guessed.
- **`HelmProbe` (MODEL → MODEL):** profiles an uncalibrated model by recording activations across every transformer block simultaneously in one pass. Run it twice — tag `probe_pos` with the first prompt, `probe_neg` with the contrasting one — then feed both tags to `HelmProbeReport`.
- **`HelmProbeReport` (tags → INT):** a per-block separation ranking; writes `ComfyUI/user/steer/probe_report.json` and outputs the middle-band peak block as a starting point for calibration — a measurement, not a validated block picker.

### Run as an app via the Comfy SDK

`app/run_helm.py` drives Helm workflows programmatically through the Comfy Developer Platform (`comfy-sdk`). The same command runs against a local ComfyUI instance (behind `comfy-api-proxy`) or your own hosted/serverless deployment — only `COMFY_BASE_URL` (and an API key) change. It does **not** run on Comfy Cloud: Cloud doesn't allow installing custom nodes, so Helm can't be present there.

```bash
pip install comfy-sdk comfy-api-proxy
comfy-api-proxy                                   # Serves Comfy API v2 on 127.0.0.1:8189 in front of ComfyUI :8188
export COMFY_BASE_URL=http://127.0.0.1:8189

python app/run_helm.py --list                     # Inspect available Helm sliders in the workflow
python app/run_helm.py --set mood=0.15 --set density=0.25 --out out
```

Your own hosted / serverless deployment:

```bash
export COMFY_BASE_URL=https://<deployment>.run.comfy.app
export COMFY_API_KEY=comfyui-...
python app/run_helm.py --workflow api/h3_t2v_helm_api.json --strength 0.2 --out out
```

Per-node control: `--workflow` selects the graph (default `api/h3_t2v_helm_api.json`), `--strength` is the global default, `--axis` overrides axis names, and a repeatable `--set AXIS=STRENGTH` targets individual `HelmApply` nodes; `--list` prints every node in graph order.

## Prior art & what's new here

Activation steering for diffusion is **not** new, and I'd rather say so than pretend otherwise:

- **CASteer** (Gaintseva et al., ICLR 2026, [arXiv:2503.09630](https://arxiv.org/abs/2503.09630)) — training-free steering vectors built by averaging cross-attention activations, for concept erasure in image diffusion.
- **Concept Sliders** (Gandikota et al., ECCV 2024, [arXiv:2311.12092](https://arxiv.org/abs/2311.12092)) — interpretable sliders, but each one is a trained LoRA adaptor.

What Helm adds is deliberately narrow and testable: no training and no adapter (a ~22 KB safetensors direction built from two passes at inference time), on a **video** DiT (MiniMax H3, not an image UNet), inside ComfyUI as a node pack with per-block calibration, a `safe_max` clamp, and self-describing axes — plus published negative results (the 78-axis sweep and the refuted predictor). I ran that comparison with a fair design (both methods from the same starting prompt) — see §4 and `docs/EXPERIMENTS.md` §E. The mix is a scene switch; Helm is a graded change at the same framing.

## Measured science (not vibes)

In one line: **78 captured, 55 dosed, 53 labelled; 10/53 hold under the pre-registered criterion and 7/53 with the post-hoc ceiling** — the "7–10 of 53" from the video.

### 1. Monotonic dose-response & seed replication

![density axis dose-response](docs/images/dose_response.png)

On MiniMax H3 (block 20, dim 5376), sweeping the density axis moves CLIP `P(empty room)` from 0.067 → 0.980 by strength 0.30, with edge density falling 0.007 → 0.003 — saturation and edge density fall monotonically while the CLIP score climbs, and the effect reproduces across seeds (seed 23 at 0.30 reads 0.990 vs seed 11's 0.980). The full table, with the honest caveats about the CLIP probe's high-strength confound, is in [`docs/EXPERIMENTS.md` §A](docs/EXPERIMENTS.md#a-dose-response-density-axis).

### 2. Direction vs. magnitude control (orthogonality check)

![same shot, two axes: density vs hue](docs/images/axis_vs_magnitude.png)

At the same `strength = 0.30`, the density axis drains saturation (HSV sat 0.073) while a hue-only axis — same composition, differing only in hue — inflates it (0.917) and preserves edges (edge density 0.009 vs 0.003). The two directions push different, specific properties: the effect is directional, not a generic magnitude wash. [`docs/EXPERIMENTS.md` §B](docs/EXPERIMENTS.md#b-is-it-a-direction-or-just-magnitude-hue-control).

### 3. Cross-architecture reproduction & the 78-axis stress test

The same contrast in a second architecture (Z-Image / Lumina2, layers 9–21 of 30): the identical empty-vs-packed contrast reproduces monotonic content removal — edge density 0.018 → 0.001. A "density" direction isn't unique to H3's architecture; this is a same-effect result in two models, not vector transfer. [`docs/EXPERIMENTS.md` §C](docs/EXPERIMENTS.md#c-the-same-contrast-in-a-second-model-partial).

![Z-Image density](docs/images/zimage_density.png)

**78 directions measured on Z-Image — and killing our own predictor.** Of 78 captured directions, 55 were dosed and 53 labelled. Under the pre-registered criterion, **10 of 53 hold in `push`** (43 wash, 2 unmeasured); a post-hoc ceiling (saturation ≤ 0.5 and edge ≤ 2× baseline) leaves **7 of 53**. The earlier "1 in 3" was a coarse summary and is superseded by the 2026-10-19 labels. The lowest survivor sits at 0.31 cosine alignment while a washed-out axis sits at 0.94 (cosine, not a separation score). At n=23 axes, cosine alignment with the mean activation appeared to predict usable axes perfectly. I scaled the test to n=78, watched the separation collapse, and killed the predictor hypothesis rather than ship a false filter — the numbers stay, the story goes. The table is in the repo: [`zimage-78-axes.json`](docs/data/zimage-78-axes.json) (one row per axis: norm, cosine, and whether it was dosed) and [`zimage-dose-response.json`](docs/data/zimage-dose-response.json) (163 dose points covering 55 of the 78 — the other 23 remain unmeasured).

### 4. Not a prompt mixer (measured, fair design)

![baseline vs Helm vs prompt mix](docs/images/bench2/strip.png)

Both methods start from the same full-room prompt and move toward empty (Z-Image-Turbo, block 15, seed 11; seed 23 replicates). At 0.5, Helm keeps the framing and grades it (luminance 0.317 → 0.472, edge 0.0266 → 0.0209); `ConditioningAverage` between 0.9 and 0.7 is visually a no-op and only flips the scene at 0.3 (`P(empty)` 0.998). The mix replaces the scene; Helm is a graded control at the same prompt. Full table and caveats: [`docs/EXPERIMENTS.md` §E](docs/EXPERIMENTS.md#e-helm-vs-prompt-mixing-conditioningaverage--fair-re-test-2026-10-19); raw numbers: [`bench2-vs-prompt-mix.json`](docs/data/bench2-vs-prompt-mix.json).

## Calibration table (`calibration.json`)

| Model | Module | Blocks | Hidden dim | Default block | Verified band | Usable strength | Axis norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniMax H3 | `blocks` | 50 | 5376 | 20 | 15–35 | 0.01–0.30 | ~3000–4000 |
| Z-Image / Lumina2 | `layers` | 30 | 3840 | 15 | 9–21 | 0.50–3.00 | ~19 |
| LTX-2.3 | `transformer_blocks` | — | — | 20 | — | — | — |

The shipped H3 axes record `safe_max: 0.3`; the shipped Z-Image room-density axis records 0.25 (its wash is visible at 0.5). `HelmApply` clamps any strength past the recorded `safe_max`. LTX-2.3 ships flag-wrapped defaults with `tested: false` — the `transformer_blocks` module is wired, and the numbers are H3-derived placeholders pending a sweep. Axis scale is model-specific; confirm every model with 1–2 renders. And the Z-Image band is not a law: on the §E benchmark axis the wash is already visible at 0.5, so check the wash onset per axis.

### Study layer

The derived study — normalised CSVs, nine charts, the 55-axis atlas, and the negative-results note — lives in [`study/`](study/README.md). It turns the JSON in `docs/data/` into tables and a browsable atlas, and keeps the failures right next to the survivors. Rebuild with `make -C study`; the source of truth stays `docs/EXPERIMENTS.md` and `docs/data/`.

## Limitations

I'd rather list what this *doesn't* do than leave you to guess:

- **Two models measured with Helm.** H3 (t2v) and Z-Image — that's where an axis, a band, and a dose were actually measured. The models used elsewhere are not Helm-measured: Krea 2 (templates wired, no calibration entry), LTX-2.3 (`tested: false`), Flux (tuple slot declared, no dose), Qwen-Image/SD3 (tuple `(txt, img)` refused until declared/measured). No Helm claim rests on them.
- **The axes are low-level directions, not concepts.** Pushed far enough they wash the frame out; the wash is the signature of leaving the model's manifold, not a feature.
- **One baseline measured.** Helm vs prompt-only (`ConditioningAverage`) is measured in §4 / `docs/EXPERIMENTS.md` §E (one model, one contrast pair, two seeds). A LoRA baseline is not measured.
- **The pre-registered criterion has no upper bound on saturation or edge density.** A post-hoc ceiling (saturation ≤ 0.5, edge ≤ 2× baseline) leaves 7/53 `push` holds; the 3 primary survivors it excludes are the saturated/specular axes at their top dose (`z_saturated_vs_muted_stock`, `z_specular_reflection_density`, `z_specular_reflection_intensity`). The primary labels are unchanged; `strict_push` is in `docs/data/axis-labels.json`.
- **Norm-preserving and projection-clamping did not help (measured).** On the 55-axis bench with a pre-registered criterion, `push` holds 10/53 axes, `norm` 1/53 and `clamp` 2/53: both alternatives collapse luminance instead of removing the wash. The step window is a partial mitigation, not a fix. Details: [`docs/EXPERIMENTS.md` §F](docs/EXPERIMENTS.md#f-fix-vs-push-and-the-step-window--measured-2026-10-19).
- **What's measured is content and geometry proxies** (CLIP empty-room score, saturation, edge density) on one contrast pair per model. Subject identity, camera movement and lighting are not measured.
- **Regional masks localize the injection, not the visible effect.** The delta is zero outside the mask (unit-tested), but attention propagates the edit: in the 2026-10-19 Z-Image demos >98% of outside pixels still moved (inside/outside mean |difference| ratio 1.4–1.7). Don't read `mask` as spatial containment.
- **Axis scale is model-specific.** The H3 wash-out onset (~0.45 at axis norm ~3000) isn't comparable to the Z-Image band (norm ~19), which is why the calibration table lists them separately.
- **Custom nodes only.** Comfy Cloud doesn't install custom nodes, so Helm runs on local or self-hosted ComfyUI.
- **A shipped artifact has broken once.** Commit `3003fc8` replaced `example_axes/mood.safetensors` with a 6-value unit-test stub (the H3 template then failed on a dimension mismatch). It's restored, `tests/test_example_axes.py` now validates every shipped axis, and `tests/conftest.py` fails the suite if anything writes into `example_axes/`.

## Architecture & stream map

Where `HelmApply` injects into a block's output changes by architecture — some blocks return plain tensors, some return tuples where the element order isn't standard:

| Architecture | Block attribute | Block output | Visual stream slot | Status in Helm |
| --- | --- | --- | --- | --- |
| MiniMax H3 | `blocks` | Tensor (packed sequence) | — | Calibrated & verified |
| Z-Image / Lumina2 | `layers` | Tensor | — | Calibrated & verified |
| LTX-2.3 | `transformer_blocks` | Tensor | — | Hook ready (`tested: false`) |
| Flux (`DoubleStreamBlock`) | `double_blocks` | Tuple `(img, txt)` | Element 0 | Declared in `_TUPLE_VISUAL_BY_CLASS` |
| SD3 | `joint_blocks` | Tuple `(txt, img)` | Element 1 | Refused until measured |
| Qwen-Image | `transformer_blocks` | Tuple `(txt, img)` | Element 1 | Refused until measured |

Zero silent guessing: an undeclared tuple block raises an explicit `RuntimeError` naming the class, rather than steering the text conditioning (SD3 and Qwen-Image would do exactly that if guessed as `out[0]`). `_BLOCK_ATTRS` and `_TUPLE_VISUAL_BY_CLASS` in `nodes.py` are where an architecture gets declared; the refusal error lists every block attribute actually present on the module.

## Engineering notes & hardening

- **Self-contained `.safetensors` axes (no pickle).** Header metadata (dim, block, model, module, norm, `safe_max`); the legacy `.pt` read path is kept for 0.2.0 axes and loads strictly with `weights_only=True`.
- **Resolution order without metadata shadowing.** `user/steer` → `models/steer` → `example_axes/` (preferring `.safetensors` over a legacy `.pt` in the same directory); metadata comes strictly from the axis's own `.safetensors` header or sidecar, so a bundled axis never shadows a user's file. A missing axis raises a `FileNotFoundError` listing every directory searched.
- **MiniMax H3 aimdo bypass.** `aimdo_enabled = False` strictly for the wrapped call, restored in the `finally` block along with hook removal.
- **77 automated tests** (73 test functions; one parametrised over five block attributes) — silent-failure modes pinned; the full defect log is [`docs/HARDENING.md`](docs/HARDENING.md).

```bash
curl -s http://127.0.0.1:8188/object_info | grep -o 'Helm[A-Za-z]*' | sort -u
# HelmApply
# HelmCollect
# HelmProbe
# HelmProbeReport
# HelmVector
```

## Licensing & credits

| Component | License |
| --- | --- |
| comfyui-helm (nodes, SDK runner, example workflows, prebuilt axes) | MIT License |
| Model weights (H3, Z-Image, LTX, Krea2, Flux) | Governed by their respective licenses; no weights bundled |

Built by **[ U N B R A N D E D ]** for the ComfyUI community.

### Related (public)

- [`comfy-api-graphs`](https://github.com/U-N-B-R-A-N-D-E-D/comfy-api-graphs) — a zero-dependency Python helper to load, mutate, and validate ComfyUI API-format graphs.

---

**About** — Training-free activation steering for ComfyUI: read a transformer block's direction between two prompts and inject it with a single strength scalar. Open weights only.

`www.kafein.agency/unbranded/helm`
