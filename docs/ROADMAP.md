# Roadmap

Status: **0.3.8** released; `main` adds `mask_offset` (PRs #12/#13) and the
2026-10-19 provenance index. 75 tests.

## Next, in order

1. **Krea 2 calibration.** The pack wires Krea 2, but it is **not validated**.
   Map its blocks, capture 1-2 axes, run a dose sweep under the same
   pre-registered criterion, and record `safe_max`. Do not claim still-image
   support before this.
2. **H3 i2v template.** The shipped H3 API graph is t2v. Add and validate an i2v
   graph; record the video-slice offset for masks (`seq - frames*h*w`).
3. **Interaction measurements.** LoRA + Helm (including the H3 continuation
   LoRA), `cfg=1`, `cond_only` on/off, cross-seed. Publish positives and
   negatives with the same provenance.
4. **Masks.** Documented as injection-only: the delta is zero outside the mask,
   but attention propagates the visible change (>98% of outside pixels move in
   the 2026-10-19 demos). Do not advertise spatial containment.
5. **0.3.9 + registry.** Bump the version, retag, publish the registry entry,
   and point the docs at the measured limits.

## Measured (2026-10-19, Z-Image, 55 axes)

- `push` 10/53 holds, `norm` 1/53, `clamp` 2/53 under the pre-registered
  criterion; norm/clamp collapse luminance.
- Step window (`start=0.6`) is a partial mitigation, not a fix.
- Z-Image: 4128 tokens = 32 packed + 4096 spatial (`mask_offset=32`).
- H3: packed `[text|cond|audio|video]`, video order `t,h,w`.
- 385/385 jobs rendered with per-render sidecars and graph hashes.

Evidence: [`EXPERIMENTS.md`](EXPERIMENTS.md) §F,
[`data/evolution-20261019/`](data/evolution-20261019/),
[`data/axis-labels.json`](data/axis-labels.json).
