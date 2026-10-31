# Roadmap

Status: **0.4.0** active in the Comfy Registry. 211 tests.

## Now — what judges see first

1. **UX release.** `Helm · Dial` and `Helm · Make Axis` are the entry points;
   three flat starter templates put the prompt, the Dial and the result on the
   canvas. The measured claim stays at **10/53** push holds and **7/53** strict.
2. **Publish 0.4.0 to the Comfy Registry.** This is the entry point to
   ComfyUI-Manager and the community.
3. **Video + social post + submission.** Server-room spine: H3 `density`
   (the measured band) and the Z-Image `specular` axis at 1.5 (clean at that
   strength). One line on why it is different: a ~22 KB direction, no training,
   no extra sampling, measured negatives published.
4. **0.5 is experimental.** The Axis / Shape / Mix / Apply chain stays out of
   the release until its UX and GPU claims are validated.

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
