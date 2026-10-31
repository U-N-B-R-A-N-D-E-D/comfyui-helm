# Roadmap

Status: **0.3.11** active in the Comfy Registry. 84 tests.

## Now — what judges see first

1. **Consistency pass.** README reports the pre-registered **10/53** push holds
   and the post-hoc **7/53** strict ceiling (saturation <= 0.5, edge <= 2x
   baseline); the Z-Image room-density axis ships `safe_max 0.25` (wash at 0.5)
   and its template uses 0.25; the grid tooltip matches the code; public docs no
   longer point at private paths.
2. **Publish 0.3.11 to the Comfy Registry.** This is the entry point to
   ComfyUI-Manager and the community; it adds the workflow axis guard, the
   axis picker and `safe_max` warnings.
3. **Video v32 + social post + submission.** Server-room spine: H3 `density`
   (the measured band) and the Z-Image `specular` axis at 1.5 (clean at that
   strength). One line on why it is different: a ~22 KB direction, no training,
   no extra sampling, measured negatives published.
4. **Submit** with `#OpenSourceHiggsfield`, ideally before the build nights on
   Oct 28-29.

## Next — research (after submission)

5. **Krea 2 calibration** (blocks + dose; gate: >= 2 axes hold, else the still
   stays without Helm).
6. **H3 i2v API template** (the pack ships t2v).
7. **Interactions and ceilings:** LoRA + Helm, `cfg=1`, `cond_only`, cross-seed,
   and a saturation/edge ceiling per model (the post-hoc check above is a first
   step, not a law).

Measured evidence: [`EXPERIMENTS.md`](EXPERIMENTS.md) §F,
[`data/axis-labels.json`](data/axis-labels.json),
[`data/evolution-20261019/`](data/evolution-20261019/).
