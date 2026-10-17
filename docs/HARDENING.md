# Hardening log

The defects behind 0.2.0, each as symptom → root cause → fix → verification. Root-cause line
references are to the pre-fix tree. Every listed test lives in `tests/` and runs headless on CPU
torch (`pytest -q`), with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The theme is silent failure: each defect produced wrong output, no output, or a crash on a path the
docs implied was supported, without a clear error.

## E1 — two chained `HelmApply` silently dropped the first slider

- **Symptom.** `h3_t2v_helm.json` chains `mood` → `density`. Only the second axis changed the output;
  moving the first slider did nothing, with no log or error.
- **Root cause.** ComfyUI's `model_function_wrapper` is a single slot.
  `nodes.py:143` (`HelmCollect`) and `nodes.py:223` (`HelmApply`) each called
  `mp.set_model_unet_function_wrapper(...)`, so the second call overwrote the first. The first hook
  was never installed.
- **Fix.** Store hooks on the patcher (`mp.model_options["helm_hooks"]`, a list) and install one
  wrapper that runs them all. `_add_hook` appends; `_wrapper` iterates the list, and if a foreign
  `model_function_wrapper` was already present it calls the previous wrapper rather than clobbering
  it. The wrapper is marked `_helm_wrapper` so a second `HelmApply` reuses it instead of nesting.
- **Verification.** `tests/test_compose.py`: two chained applies produce output that differs from the
  base and from either single-axis result; a foreign wrapper is preserved (asserted called).

## E2 — fresh clone crashed with `FileNotFoundError`

- **Symptom.** A clean clone with the shipped `example_axes/mood.pt` and `density.pt` raised
  `FileNotFoundError` on the first `HelmApply`, despite the README saying the prebuilt axes worked.
- **Root cause.** The code read and wrote the single path `user/steer` (`nodes.py:24`, `:33`–`:38`),
  while the docs said axes lived in `models/steer`, and the prebuilt axes actually shipped in
  `example_axes/`. None of the three agreed.
- **Fix.** Writes go to `folder_paths.get_user_directory()/steer`. Reads search, in order,
  `user/steer/` → `models/steer/` → the pack's own `example_axes/`; `_resolve_axis` / `_resolve_meta`
  return the first existing match, and the missing-file error lists every directory searched plus how
  to fix it.
- **Verification.** `tests/test_axes.py`: resolution order, and the missing-axis error listing all
  dirs.

## E3 — the shipped LTX workflow could not run

- **Symptom.** `ltx_t2v_helm.json` failed at `_find_blocks` even though the model exposes its
  transformer blocks.
- **Root cause.** `nodes.py:79`–`:89` (`_find_blocks`) only tried `blocks` and `layers`. LTX names its
  blocks `transformer_blocks`, so discovery raised `RuntimeError`.
- **Fix.** `_BLOCK_ATTRS = ["blocks", "layers", "transformer_blocks", "double_blocks",
  "single_blocks"]`; `_find_blocks` returns `(modulelist, attr_name)` and the error names the attrs
  actually present. `calibration.json` maps the LTX entry `module` to `"transformer_blocks"`.
- **Verification.** `tests/test_blocks.py`: each attribute is found on a fake module; the error lists
  the present attrs.

## E4 — sidecar `"created"` was written as `null`

- **Symptom.** Every axis `.json` sidecar recorded `"created": null` instead of a timestamp.
- **Root cause.** `nodes.py:41`–`:48` (`_write_meta`) unconditionally assigned
  `meta["created"] = None`, stomping the value the caller had passed in.
- **Fix.** `setdefault` for both fields: `meta.setdefault("helm_version", 2)` and
  `meta.setdefault("created", time.time())`. A caller-supplied `created` is never overwritten. Every
  sidecar also carries `"model"` and `"module"`.
- **Verification.** `tests/test_axes.py`: `_write_meta` preserves a supplied `created` and stamps
  `helm_version == 2`.

## E5 — `calibration.json` was dead code

- **Symptom.** `calibration.json` listed per-model `module`, `default_block` and strength bands, but
  nothing read them; every node used a hardcoded block 20 regardless of model.
- **Root cause.** `_calibration()` (`nodes.py:59`–`:64`) was defined and never called. The `block`
  default was hardcoded at `nodes.py:122` and `:184`, and `_find_blocks` was always called without a
  module override.
- **Fix.** `_calib_for(mp)` matches the `diffusion_model` class name against each entry's `match`
  substrings. A match supplies the module for `_find_blocks`, and the matched module is logged on
  each run. `INPUT_TYPES` stays static, so the frozen widget order is unchanged, and the `block`
  default remains 20.
- **Verification.** `tests/test_calibration.py`: `_calib_for` matches `MiniMaxH3` to the H3 entry,
  returns `None` for unknown models, and `calibration.json`'s LTX `module == "transformer_blocks"`.

## E6 — `app/run_helm.py` applied one strength to every slider

- **Symptom.** The runner set the same `strength` on every `HelmApply` node, so a graph with two axes
  could not be driven with per-axis strengths, even though the axes have different scales.
- **Root cause.** `app/run_helm.py:37`–`:43` looped over every `HelmApply` and called
  `wf.set_input(node_id, "strength", args.strength)` unconditionally.
- **Fix.** Added `--set axis=strength` (repeatable), matched by the node's current axis name, with
  position as the fallback; `--list` prints `index  axis  strength  [block]` per node. Precedence:
  `--set` entry > global `--strength`, with `--axis` overriding the name only for unmatched nodes.
  Unknown `--set` names exit with the list of nodes found.
- **Verification.** `tests/test_run_helm.py`: `--set` / `--list` parsing against
  `api/h3_t2v_helm_api.json`.

## Collect write policy — one write per run

- **Symptom.** `HelmCollect` saved the accumulating axis to disk on **every** sampling step (a
  `torch.save` per step per run), which is needless I/O and leaves a partial axis on disk mid-run.
- **Root cause.** `nodes.py:138` called `torch.save` inside the forward hook, which fires once per
  block per step.
- **Fix.** Accumulate in memory and write once per run — on the last step when the schedule length is
  known, otherwise every step as a fallback. The accumulator resets at `step == 0`, so runs do not
  bleed into each other.
- **Verification.** `tests/test_collect.py`: a write counter shows one write per run, and running
  twice does not accumulate (`n` resets).

## New knobs — `cond_only` and windowing

Not defects; defaults preserve today's behavior. Added with the E1 hook rework.

- **`cond_only`** (`HelmCollect`, `HelmApply`; default `False`). Collection and injection can be
  restricted to conditional batch items (`cond_or_uncond` flag == 0); `_mask_by_flags` zeroes the rest
  and falls back to no masking if the flag list does not line up. The recommended setting for CFG
  models; the shipped axes were not measured with it on, so it is **not yet measured**.
- **Windowing** (`HelmApply` `start_percent` / `end_percent`; default `0.0` / `1.0`). The composing
  wrapper skips a hook when the sampling fraction `step / total` falls outside `[start, end]` using
  `transformer_options["sample_sigmas"]`. The default window is the whole run; the knob is
  **not yet measured**.
- **Verification.** `tests/test_cond.py` (uncond batch items masked, collect and apply),
  `tests/test_window.py` (hooks outside `[start, end]` skipped).
- **Open item.** A dose-response re-measure on the pod, with these knobs on, is pending; until then
  the results in [`EXPERIMENTS.md`](EXPERIMENTS.md) hold only for the defaults.
