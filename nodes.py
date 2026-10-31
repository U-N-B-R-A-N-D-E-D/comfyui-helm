"""Helm — training-free activation steering for ComfyUI.

Read a transformer block's mean activation for two contrastive prompts (diff-in-means) and inject
`x + strength*axis` at that block during sampling. Weights are not modified; an axis is a small
`.safetensors` file that carries its own metadata (dim, block, model). Reads resolve `user/steer/` ->
`models/steer/` -> this pack's `example_axes/`; writes always go to `user/steer/`. Axes written by
older versions as `.pt` are still read; `HelmApply` `block=-1` adopts the block recorded in the axis.

Everyday nodes (category `Helm`)
  HelmDial      MODEL -> MODEL   pick an idea by name, move Amount 0-1. No strength, norm or block to know:
                                 amount maps to 0..safe_max recorded inside the axis file.
  HelmMakeAxis  MODEL+CLIP -> MODEL, STRING   write "with the idea" / "without the idea", press Run:
                                 records both contrast passes in one execution, saves the axis file.

Lab nodes (category `Helm/Lab`, class names unchanged so old workflows still load)
  HelmCollect  MODEL -> MODEL   mean activation at block K for a tag (run once per prompt)
  HelmVector   tags  -> STRING  axis = mean_pos - mean_neg -> reusable axis file
  HelmApply    MODEL -> MODEL   inject x + strength*axis at block K (chain several for multiple axes)

Model onboarding
  HelmProbe        MODEL -> MODEL  read EVERY block for a tag (run twice: pos, then neg)
  HelmProbeReport  tags  -> INT    per-block separation ranking -> user/steer/probe_report.json, plus
                                   the highest-separation block in the middle 50%. That peak is a
                                   measurement, not a validated picker (docs/EXPERIMENTS.md D)

Wire MODEL source -> HelmApply -> sampler. Works on any DiT exposing a `blocks`/`layers`/`transformer_blocks`/
`double_blocks`/`single_blocks` ModuleList (H3: blocks, dim 5376, block ~20; Z-Image/Lumina: layers,
dim 3840, layers 9-21). Per-model defaults are in calibration.json; the axis scale is model-specific.
"""
import json
import os
import time
from functools import lru_cache, partial

import torch
import folder_paths
from safetensors import safe_open as _st_open
from safetensors.torch import save_file as _st_save

# --- paths / calibration -------------------------------------------------------------------
_PACK_DIR = os.path.dirname(os.path.realpath(__file__))
AXIS_DIR = os.path.join(folder_paths.get_user_directory(), "steer")
_READ_DIRS = [AXIS_DIR, os.path.join(folder_paths.models_dir, "steer"), os.path.join(_PACK_DIR, "example_axes")]
_CALIB_PATH = os.path.join(_PACK_DIR, "calibration.json")

# Axes are `.safetensors` — self-contained (tensor + metadata in the header), no pickle anywhere.
# `.pt` is still READ so axes written by 0.2.0 keep working; nothing writes it anymore.
AXIS_EXT = ".safetensors"
LEGACY_EXT = ".pt"


def _axes():
    os.makedirs(AXIS_DIR, exist_ok=True)
    return AXIS_DIR


def _axis_path(name):
    os.makedirs(AXIS_DIR, exist_ok=True)
    return os.path.join(AXIS_DIR, f"{name}{AXIS_EXT}")


def _meta_path(name):
    return os.path.join(AXIS_DIR, f"{name}.json")


def _resolve_axis(name):
    """An axis name resolves per directory (read order preserved), preferring `.safetensors` over a
    legacy `.pt` inside the same directory, so a new bundled axis never shadows a user's own file."""
    tried = []
    for d in _READ_DIRS:
        for ext in (AXIS_EXT, LEGACY_EXT):
            p = os.path.join(d, f"{name}{ext}")
            tried.append(p)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"axis '{name}' not found. Looked in:\n  " + "\n  ".join(tried) +
        f"\nRun HelmCollect + HelmVector first, or drop '{name}{AXIS_EXT}' into {_READ_DIRS[-1]}.")


def _resolve_meta(name):
    tried = []
    for d in _READ_DIRS:
        p = os.path.join(d, f"{name}.json")
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"metadata for '{name}' not found. Looked in:\n  " + "\n  ".join(tried) +
        f"\nIt is written beside the axis by HelmCollect/HelmVector.")


def _load_tensor(path, key="axis"):
    """`[dim]` or `[n_blocks, dim]` float32. `.safetensors` needs no pickle; a legacy `.pt` is loaded
    with `weights_only=True`, so a dropped axis file can never execute code."""
    if path.endswith(AXIS_EXT):
        with _st_open(path, framework="pt", device="cpu") as f:
            vec = f.get_tensor(key)
    else:
        vec = torch.load(path, map_location="cpu", weights_only=True)
    return vec.detach().to(torch.float32)


def _load_axis(path):
    return _load_tensor(path).reshape(-1)


def _meta_from_header(header):
    """Pure. safetensors metadata values are strings, so cast back; `''` is the encoding of None."""
    out = {}
    for k, v in header.items():
        if k in ("dim", "block"):
            try:
                out[k] = int(v)
                continue
            except (TypeError, ValueError):
                pass
        elif k in ("norm", "safe_max"):
            try:
                out[k] = float(v)
                continue
            except (TypeError, ValueError):
                pass
        elif k in ("from", "probe"):
            try:
                out[k] = json.loads(v)
                continue
            except (TypeError, ValueError):
                pass
        out[k] = v or None
    return out


def _load_axis_meta(path):
    try:
        with _st_open(path, framework="pt", device="cpu") as f:
            return _meta_from_header(f.metadata() or {})
    except Exception:
        return None


def _save_tensor(name, tensor, meta, key="axis"):
    """Write a tensor and its metadata as ONE `.safetensors` file. The JSON sidecar is kept too --
    it is greppable and hand-editable -- but the file is complete without it."""
    os.makedirs(AXIS_DIR, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("helm_version", 2)
    meta.setdefault("created", time.time())
    meta.setdefault("model", None)
    meta.setdefault("module", None)
    header = {k: (json.dumps(v) if isinstance(v, (dict, list)) else "" if v is None else str(v))
              for k, v in meta.items()}
    _st_save({key: tensor.detach().to(torch.float32).cpu().contiguous()}, _axis_path(name), metadata=header)
    _write_meta(name, meta)


def _save_axis(name, vec, meta):
    _save_tensor(name, vec.reshape(-1), meta)


def _write_meta(name, meta):
    try:
        meta.setdefault("helm_version", 2)
        meta.setdefault("created", time.time())
        meta.setdefault("model", None)
        meta.setdefault("module", None)
        with open(_meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[helm] could not write metadata for '{name}': {e}")


def _read_meta(name):
    # D2: a sidecar from another directory must never describe this axis. Look beside the resolved
    # axis first, then the header -- only then fall back to a name match anywhere.
    try:
        axis_file = _resolve_axis(name)
    except Exception:
        axis_file = None
    if axis_file:
        sidecar = os.path.splitext(axis_file)[0] + ".json"
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
        meta = _load_axis_meta(axis_file)
        if meta:
            return meta
        # A legacy .pt with no sidecar beside it simply has no metadata -- it must never inherit a
        # same-named sidecar from another directory (audit 4.1.1).
        return None
    # No axis file at all: a meta-only sidecar (written by _write_meta) is the only thing that can
    # describe this name, and it is genuinely its own.
    try:
        with open(_resolve_meta(name), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _calibration():
    try:
        with open(_CALIB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"models": []}


def _calib_for(mp):
    try:
        cand = [mp.model.diffusion_model.__class__.__name__.lower()]
    except Exception:
        return None
    try:
        extra = str(getattr(mp.model, "model_name", "") or "").lower()
        if extra:
            cand.append(extra)
    except Exception:
        pass
    for entry in _calibration().get("models", []):
        for m in entry.get("match", []):
            if any(str(m).lower() in c for c in cand):
                return entry
    return None


def _model_name(mp):
    try:
        return mp.model.diffusion_model.__class__.__name__
    except Exception:
        return None


# --- hook machinery ------------------------------------------------------------------------
def _stack_mean(m, o, flags=None, cond_only=False):
    # D1: reading must go through the same slot rule as injection. Reading out[0] of a tuple would
    # silently average the TEXT stream on architectures that return (txt, img).
    slot = _tuple_slot(m, o)
    x = (o[slot] if slot is not None else o).detach().float()
    if cond_only:
        if flags is None:
            raise RuntimeError("cond_only=True but this model reported no cond_or_uncond flags; "
                               "set cond_only=False (nothing was masked, and saying so beats guessing).")
        f = torch.as_tensor(flags).reshape(-1)
        if f.numel() != x.shape[0]:
            raise RuntimeError(f"cond_only=True but {f.numel()} flags do not match the batch "
                               f"({x.shape[0]}); refusing to average cond and uncond together.")
        sel = x[f.to(x.device) == 0]
        if sel.shape[0] == 0:
            raise RuntimeError("cond_only=True but no unconditional item is present in this batch.")
        x = sel
    while x.dim() > 1:
        x = x.mean(dim=0)
    return x  # -> [dim]


def _mask_by_flags(d, x, flags):
    try:
        f = torch.as_tensor(flags).reshape(-1)
        if f.numel() != x.shape[0]:
            return d
        m = (f.to(x.device) == 0).to(dtype=d.dtype, device=d.device)
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        return d * m
    except Exception:
        return d


_BLOCK_ATTRS = ["blocks", "layers", "transformer_blocks", "joint_blocks", "double_blocks",
                "single_blocks"]


def _find_blocks(model, which=None):
    dm = model.model.diffusion_model
    names = [which] if which else list(_BLOCK_ATTRS)
    for n in names:
        mod = getattr(dm, n, None)
        if mod is not None and len(mod) > 4:
            return mod, n
    present = [a for a in _BLOCK_ATTRS if getattr(dm, a, None) is not None]
    raise RuntimeError(
        f"no transformer block list on {dm.__class__.__name__} (looked for {names}; "
        f"module has {present or 'none of the known ones'}). Pass the attribute via calibration.json.")


def _flush(tag, acc, block, module, model_name):
    if acc.get("sum") is None or acc.get("n", 0) <= 0:
        return
    v = acc["sum"] / acc["n"]
    _save_axis(tag, v, {"kind": "activation", "dim": int(v.shape[0]), "block": int(block),
                        "model": model_name, "module": module})


def _make_collect_hook(acc, tag, block, module, model_name, cond_only, write_mode):
    def fn(m, args, out, flags=None):
        v = _stack_mean(m, out, flags=flags, cond_only=cond_only)
        acc["sum"] = v if acc["sum"] is None else acc["sum"] + v
        acc["n"] += 1
        if write_mode == "each" or (write_mode == "last" and acc.get("last")):
            _flush(tag, acc, block, module, model_name)
        return None  # read-only
    fn._acc = acc
    return fn


# A block that returns a tuple must say which element is the visual stream. Assuming `out[0]` silently
# steers the TEXT conditioning on any architecture that returns `(txt, img)` -- SD3 (joint_blocks) and
# Qwen-Image do exactly that. Verified against the runtime we hook: comfy/ldm/flux/layers.py:267 returns
# `(img, txt)`, comfy/ldm/qwen_image/model.py returns `(txt, img)`. Per-architecture table:
# README section 'Supported architectures', which carries the same table.
_TUPLE_VISUAL_BY_CLASS = {"DoubleStreamBlock": 0}


def _tuple_slot(module, out):
    """Index of the visual stream in a tuple return, or None when the block returns a plain tensor."""
    if not isinstance(out, tuple):
        return None
    name = type(module).__name__
    if name in _TUPLE_VISUAL_BY_CLASS:
        return _TUPLE_VISUAL_BY_CLASS[name]
    raise RuntimeError(
        f"block '{name}' returns a tuple and Helm does not know which element is the visual stream. "
        f"Steering the wrong one would move the text conditioning instead of the image, so this is "
        f"refused rather than guessed. Declare it in _TUPLE_VISUAL_BY_CLASS (the per-architecture table "
        f"lives in the README's Supported architectures section).")


def _steer(x, d, u, mode="push", clamp_k=1.0):
    """Pure. Apply one Helm step to the visual-stream tensor.

    push  : x + d                      (the published behavior; default)
    norm  : x + d, then rescale each token to ||x|| -- the wash looks like excess energy
    clamp : move the projection onto the axis and clamp it to +/- clamp_k*||x||
            ("fix instead of push": never leave the range the model already produced)
    """
    if mode not in ("push", "norm", "clamp"):
        raise ValueError(f"unknown Helm mode {mode!r}; expected push, norm or clamp")
    if torch.count_nonzero(d) == 0:
        return x
    if mode == "push":
        return x + d
    if mode == "norm":
        y = x + d
        nx = x.norm(dim=-1, keepdim=True)
        ny = y.norm(dim=-1, keepdim=True)
        return y * (nx / (ny + 1e-8))
    if mode == "clamp":
        p = (x * u).sum(dim=-1, keepdim=True)
        delta = (d * u).sum(dim=-1, keepdim=True)
        lim = float(clamp_k) * x.norm(dim=-1, keepdim=True)
        p_new = torch.clamp(p + delta, -lim, lim)
        return x + (p_new - p) * u


def _parse_grid(grid):
    """Pure. 'H,W' or 'H,W,T'. Explicit only: a mask never infers the layout."""
    parts = [int(v) for v in grid.replace("x", ",").replace(" ", ",").split(",") if v.strip()]
    if len(parts) == 2:
        return parts[0], parts[1], 1
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"grid must be H,W or H,W,T (got {grid!r})")


def _token_mask(mask2d, layout, frames):
    """Pure. mask2d: [h, w] (bool/float). Returns [1, h*w*frames, 1] in the given token order."""
    m = torch.as_tensor(mask2d, dtype=torch.float32).reshape(-1)
    if layout == "h_w":
        flat = m
    elif layout == "t_h_w":                      # t outer, then h, then w
        flat = m.repeat(int(frames))
    elif layout == "h_w_t":                      # h,w outer, t inner
        flat = m.reshape(-1, 1).repeat(1, int(frames)).reshape(-1)
    else:
        raise ValueError(f"unknown mask layout {layout!r}; expected h_w, t_h_w or h_w_t")
    return flat.reshape(1, flat.numel(), 1)


def _load_mask(path, h, w):
    """Lazy Pillow import (ComfyUI ships it); resizes to the token grid and thresholds at 0.5."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("L").resize((int(w), int(h)), Image.BILINEAR)
    return np.asarray(im, dtype="float32") / 255.0 > 0.5


def _apply_token_mask(d, x, tm, offset=0):
    """Pure. Place a [n] token mask after `offset` leading tokens and apply it to the delta.

    Z-Image packs 32 extra rows before the 4096 spatial tokens; H3 packs
    [text | cond | audio | video], so a video mask needs the offset of the video slice.
    """
    seq = x.shape[1] if x.dim() == 3 else x.shape[0]
    n = int(tm.shape[1])
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"mask_offset must be >= 0 (got {offset})")
    if offset + n != seq:
        raise RuntimeError(
            f"mask covers {n} tokens + offset {offset} = {offset + n}, block has {seq}; "
            f"grid/layout/mask_offset does not match this model")
    full = torch.zeros(seq, 1, dtype=tm.dtype, device=tm.device)
    full[offset:offset + n] = tm.reshape(n, 1)
    if x.dim() == 3:
        full = full.unsqueeze(0)
    return d * full.to(device=x.device, dtype=x.dtype)


def _make_apply_hook(vec, strength, axis_name, cond_only, meta_path,
                     mode="push", token_mask=None, clamp_k=1.0, debug=False, mask_offset=0):
    unit = None
    if mode == "clamp":
        v = vec / (vec.norm() + 1e-8)
        unit = v
    logged = {"once": False}
    def fn(m, args, out, flags=None):
        slot = _tuple_slot(m, out)
        x = out[slot] if slot is not None else out
        if x.shape[-1] != vec.shape[-1]:
            raise RuntimeError(
                f"axis '{axis_name}' dim {vec.shape[-1]} != block output dim {x.shape[-1]}; "
                f"this axis was built for a different model (see {meta_path})")
        if debug and not logged["once"]:
            print(f"[helm] shape debug: block={type(m).__name__} out={tuple(out.shape) if hasattr(out, 'shape') else 'tuple'} "
                  f"x={tuple(x.shape)} seq={int(x.shape[1]) if x.dim() == 3 else int(x.shape[0]) if x.dim() == 2 else '?'} "
                  f"dim={int(x.shape[-1])} mode={mode} mask_offset={mask_offset}")
            logged["once"] = True
        d = vec.to(x.dtype).to(x.device) * strength
        if cond_only and flags is not None:
            d = _mask_by_flags(d, x, flags)
        if token_mask is not None:
            m2d, mlayout, mframes = token_mask
            tm = _token_mask(m2d, mlayout, mframes)
            d = _apply_token_mask(d, x, tm, mask_offset)
        u = unit.to(x.dtype).to(x.device) if unit is not None else None
        y = _steer(x, d, u, mode=mode, clamp_k=clamp_k)
        if slot is None:
            return y
        return tuple(y if i == slot else o for i, o in enumerate(out))
    return fn


_HELM_HOOKS = "helm_hooks"


def _add_hook(mp, hook):
    mp.model_options.setdefault(_HELM_HOOKS, []).append(hook)


def _wrapper(mp):
    """ComfyUI calls `model_function_wrapper(apply_model, {"input","timestep","c","cond_or_uncond"})`.
    All helm hooks live on the patcher (model_options['helm_hooks']) and ONE wrapper runs them, so
    chained HelmApply nodes compose instead of overwriting each other. Aimdo (H3's per-block prefetch /
    comfy_aimdo malloc graph) raises 'aimdo memory compile error' when a block's output is modified, so
    disable it for the call and restore it after."""
    prev = mp.model_options.get("model_function_wrapper")
    if getattr(prev, "_helm_wrapper", False):
        prev = getattr(prev, "_helm_prev", None)
    state = {"step": 0}

    def run(apply_model, c_in):
        import comfy.memory_management as cm
        to = c_in.get("c", {}).get("transformer_options", {}) or {}
        sig = to.get("sample_sigmas")
        total = (len(sig) - 1) if sig is not None else None
        if total and state["step"] >= total:
            state["step"] = 0
        step = state["step"]
        state["step"] += 1
        flags = c_in.get("cond_or_uncond")
        hooks = mp.model_options.get(_HELM_HOOKS) or []
        entry = _calib_for(mp)
        blocks, _ = _find_blocks(mp, which=entry.get("module") if entry else None)
        frac = (step / total) if total else 0.0
        prev_aimdo = getattr(cm, "aimdo_enabled", None)
        handles = []
        try:
            for h in hooks:
                acc = getattr(h.get("fn"), "_acc", None)
                if acc is not None:
                    if total is not None and step == 0:
                        if acc.get("reset"):
                            acc["reset"]()
                        else:
                            acc["sum"] = None
                            acc["n"] = 0
                    # SPEC DEVIATION: spec wrote `total is not None and ...`, which never writes when
                    # sample_sigmas is absent; use the spec's fallback ("each") in that case.
                    acc["last"] = (total is None) or (step + 1 >= total)
                if frac < h.get("start", 0.0) or frac > h.get("end", 1.0):
                    continue
                if h.get("all_blocks"):
                    for i in range(len(blocks)):
                        handles.append(blocks[i].register_forward_hook(partial(h["fn"], flags=flags, idx=i)))
                    continue
                idx = h["block"]
                if idx >= len(blocks):
                    raise RuntimeError(
                        f"block {idx} >= {len(blocks)} on {mp.model.diffusion_model.__class__.__name__}")
                handles.append(blocks[idx].register_forward_hook(partial(h["fn"], flags=flags)))
            if prev_aimdo is not None:
                cm.aimdo_enabled = False
            if prev is not None:
                return prev(apply_model, c_in)
            return apply_model(c_in["input"], c_in["timestep"], **c_in["c"], cond_or_uncond=c_in["cond_or_uncond"])
        finally:
            for hd in handles:
                hd.remove()
            if prev_aimdo is not None:
                cm.aimdo_enabled = prev_aimdo

    run._helm_wrapper = True
    run._helm_prev = prev
    return run


def _install_wrapper(mp):
    # SPEC DEVIATION: ComfyUI's ModelPatcher.clone() copies `model_options` with deepcopy_list_dict, so
    # a wrapper captured by an upstream HelmApply reads a stale `helm_hooks` list. Re-install on every
    # node but carry the ORIGINAL foreign wrapper forward via `_helm_prev`, so the final wrapper sees
    # every hook exactly once and any pre-existing wrapper is still chained.
    run = _wrapper(mp)
    mp.set_model_unet_function_wrapper(run)
    return run


# --- probe: per-block sensitivity profile (onboarding a new model) --------------------------
# The probe MEASURES how far two contrast prompts separate, block by block. That profile is data.
# What it does NOT do is pick the block for you: measured on Z-Image, the separation peak is block 7
# while the band verified by rendering is 9-21, and the global peak is block 0 (which destroys the
# frame). We also tried a strength estimate from the norms and it was ~10x off there, so there is no
# strength suggestion. See docs/EXPERIMENTS.md section D.


def _probe_profile(pos, neg):
    """Pure. `pos`/`neg` are `[n_blocks, dim]` per-block means. The separation is scale-free --
    `||mu_pos - mu_neg|| / mean(||mu_pos||, ||mu_neg||)` -- so blocks are comparable to each other."""
    d = pos - neg
    dn = d.norm(dim=1)
    pn, nn = pos.norm(dim=1), neg.norm(dim=1)
    mu = 0.5 * (pn + nn)
    sep = dn / (mu + 1e-8)
    cos = (pos * neg).sum(dim=1) / (pn * nn + 1e-8)
    n = int(pos.shape[0])
    lo = int(0.25 * n)
    hi = max(int(0.75 * n), lo + 1)
    k = int(lo + int(torch.argmax(sep[lo:hi])))
    return {
        "n_blocks": n,
        "dim": int(pos.shape[1]),
        "peak": {"block": k, "separation": round(float(sep[k]), 4)},
        "ranking": [{"block": int(i), "separation": round(float(sep[i]), 4),
                     "cosine": round(float(cos[i]), 4), "mu_norm": round(float(mu[i]), 2),
                     "delta_norm": round(float(dn[i]), 2)}
                    for i in torch.argsort(sep, descending=True).tolist()],
    }


def _flush_probe(tag, acc, module, model_name):
    rows, steps = acc.get("rows"), acc.get("steps", 0)
    if rows is None or steps <= 0:
        return
    prof = rows / steps
    _save_tensor(tag, prof, {"kind": "probe", "n_blocks": int(prof.shape[0]),
                             "dim": int(prof.shape[1]), "module": module, "model": model_name})


def _make_probe_hook(acc, tag, module, model_name, cond_only, write_mode):
    def fn(m, args, out, flags=None, idx=0):
        v = _stack_mean(m, out, flags=flags, cond_only=cond_only)
        if acc["rows"] is None:
            acc["rows"] = torch.zeros(acc["n_blocks"], v.shape[0], dtype=v.dtype, device=v.device)
        if idx < acc["rows"].shape[0] and v.shape[0] == acc["rows"].shape[1]:
            acc["rows"][idx] += v
        if idx == 0:
            acc["steps"] += 1
        if idx == acc["n_blocks"] - 1 and (write_mode == "each" or (write_mode == "last" and acc.get("last"))):
            _flush_probe(tag, acc, module, model_name)
        return None  # read-only
    fn._acc = acc
    return fn


# --- the Dial: human axis names, amount 0-1 -----------------------------------------------
# `axes_catalog.json` lists the axes the repo's own benchmark says HOLD (docs/EXPERIMENTS.md, study/).
# Only those are offered, and only when the axis file is actually on disk: "Essentials" (clearly visible
# change, ordered by visible impact) first, then "Subtle" (holds in the benchmark, little visible change in our
# renders). Anything else found in the read dirs (your own axes, the shipped H3 `mood`) is offered under "Custom".
_CATALOG_PATH = os.path.join(_PACK_DIR, "axes_catalog.json")
ESSENTIALS = "Essentials"
SUBTLE = "Subtle"
CUSTOM = "Custom"
# Shown first when you add a Dial: a plain, easy-to-read idea. Falls back to the first menu entry.
_DIAL_DEFAULT_FILE = "z_raking_light_angle"
_SEP = " · "
_NO_AXES = "(no axes yet - build one with Helm · Make Axis)"
_MODEL_FAMILY = {"NextDiT": "Z-Image", "MiniMaxH3Model": "MiniMax H3"}
# Make Axis stores a conservative `safe_max` when it cannot verify one. Calibrated on the single measured
# Z-Image reference: axis `z_room_density` held at safe_max 0.25, which is ~0.7% of the block-15 activation
# norm (|mu| ~964, docs/EXPERIMENTS.md D/E); the wash starts at about twice that. 0.5% sits below it.
SAFE_MAX_ACTIVATION_FRACTION = 0.005
SAFE_MAX_CEILING = 10.0


@lru_cache(maxsize=1)
def _catalog():
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return {a["file"]: a for a in json.load(f).get("axes", [])}
    except Exception:
        return {}


def _family(model_class):
    return _MODEL_FAMILY.get(model_class, model_class or "unknown model")


def _axis_names():
    """Axis files visible in the read dirs, first directory wins (same rule as `_resolve_axis`)."""
    seen, names = set(), []
    for d in _READ_DIRS:
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for ext in (AXIS_EXT, LEGACY_EXT):
            for fn in entries:
                if fn.endswith(ext) and fn[: -len(ext)] not in seen:
                    seen.add(fn[: -len(ext)])
                    names.append(fn[: -len(ext)])
    return names


def _dial_entries():
    """Pure-ish (reads the axis dirs). The Dial's menu: Essentials, then Subtle (catalog order), then Custom."""
    cat = _catalog()
    present = {}
    for name in _axis_names():
        meta = _read_meta(name) or {}
        if meta.get("kind") in ("activation", "probe", "probe_report"):
            continue  # recordings, not axes
        present[name] = meta
    entries = []
    for name, c in cat.items():
        meta = present.pop(name, None)
        if meta is None:
            continue
        fam = c.get("family") or _family(c.get("model_class"))
        section = SUBTLE if c.get("tier") == "subtle" else ESSENTIALS
        entries.append({
            "choice": f"{section}{_SEP}{c['label']} ({fam})", "label": c["label"], "section": section,
            "file": name, "family": fam, "model": meta.get("model") or c.get("model_class"),
            "safe_max": meta.get("safe_max") or c.get("fallback_safe_max"), "holds": True,
            "evidence": c.get("evidence"),
        })
    entries.sort(key=lambda e: e["section"] == SUBTLE)  # stable: Essentials keep catalog order, Subtle follow
    for name in sorted(present):
        meta = present[name]
        fam = _family(meta.get("model")) if meta.get("model") else None
        entries.append({
            "choice": f"{CUSTOM}{_SEP}{name}", "label": name, "section": CUSTOM, "file": name,
            "family": fam, "model": meta.get("model"), "safe_max": meta.get("safe_max"), "holds": None,
            "evidence": None,
        })
    return entries


def _dial_choices():
    return [e["choice"] for e in _dial_entries()] or [_NO_AXES]


def _dial_default(choices):
    for e in _dial_entries():
        if e["file"] == _DIAL_DEFAULT_FILE and e["choice"] in choices:
            return e["choice"]
    return choices[0]


def _dial_words(label, amount):
    """Pure. The one-line plain-words summary of what the Dial is doing: 'Light angle (raking) at 60%'."""
    pct = int(round(min(1.0, max(0.0, float(amount))) * 100))
    return f"{label} at {pct}%" + (" (Dial off)" if pct == 0 else "")


def _dial_resolve(text):
    """Menu text (or a bare axis name from Make Axis) -> entry dict with `file`, `label`, `safe_max`."""
    text = (text or "").strip()
    if not text or text == _NO_AXES:
        raise ValueError("No axis selected and none found yet. Build one with 'Helm · Make Axis' "
                         "(template: Helm 3 · Make your own axis), or drop an axis file into "
                         f"{AXIS_DIR}.")
    for e in _dial_entries():
        if text in (e["choice"], e["file"]):
            return e
    bare = text.split(_SEP, 1)[1] if _SEP in text else text
    for e in _dial_entries():
        if bare in (e["label"], e["file"]):
            return e
    c = _catalog().get(bare) or next(
        (a for a in _catalog().values()
         if bare in (a["label"], f"{a['label']} ({a.get('family') or _family(a.get('model_class'))})")), None)
    if c is not None:
        bare = c["file"]
        raise FileNotFoundError(
            f"The axis '{c['label']}' is part of Helm's Essentials list but its file '{bare}{AXIS_EXT}' is "
            f"not installed. Put it in {AXIS_DIR} (or build your own with 'Helm · Make Axis').")
    raise FileNotFoundError(f"axis '{bare}' not found. Pick one from the Dial menu, or build it with "
                            f"'Helm · Make Axis' first.")


def _dial_strength(amount, safe_max):
    """Pure. Amount 0..1 -> strength 0..safe_max. Out-of-range amounts are clamped, never extrapolated."""
    if not safe_max or float(safe_max) <= 0:
        raise ValueError("this axis records no safe range, so the Dial cannot scale it. Rebuild it with "
                         "'Helm · Make Axis', or use 'Helm Lab · Apply axis' with a strength.")
    a = min(1.0, max(0.0, float(amount)))
    return a * float(safe_max)


def _check_axis_fits_model(mp, entry, meta, vec):
    """Plain-English refusal when the axis was made for another model than the one connected."""
    loaded = _model_name(mp)
    made_for = meta.get("model") or entry.get("model")
    if made_for and loaded and made_for != loaded:
        raise ValueError(
            f"'{entry['label']}' was made for {_family(made_for)}, but the model connected to this Dial is "
            f"{_family(loaded)}. Pick an axis marked ({_family(loaded)}) in the menu, or build one for this "
            f"model with 'Helm · Make Axis'.")
    calib = _calib_for(mp)
    if calib and calib.get("dim") and int(calib["dim"]) != int(vec.shape[-1]):
        raise ValueError(
            f"'{entry['label']}' does not fit this model (axis width {int(vec.shape[-1])}, model width "
            f"{int(calib['dim'])}). Pick an axis made for {_family(loaded)}, or build one with "
            f"'Helm · Make Axis'.")


# --- Make Axis: both contrast passes in one execution -----------------------------------------
def _safe_axis_name(name):
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name or "").strip()).strip("_")
    if not out:
        raise ValueError("give the axis a name (letters, numbers, - and _)")
    return out


def _axis_from_means(pos_mean, neg_mean):
    """Pure. Diff-in-means: the same arithmetic as HelmVector (`mean_pos - mean_neg`)."""
    if pos_mean.shape != neg_mean.shape:
        raise ValueError(f"the two passes produced different shapes {tuple(pos_mean.shape)} vs "
                         f"{tuple(neg_mean.shape)}")
    return pos_mean.to(torch.float32) - neg_mean.to(torch.float32)


def _estimate_safe_max(pos_mean, neg_mean, fraction=SAFE_MAX_ACTIVATION_FRACTION):
    """Pure. A conservative, UNVERIFIED ceiling: the strength at which `strength*axis` is `fraction` of the
    mean activation norm (see SAFE_MAX_ACTIVATION_FRACTION). It is an estimate, not a measurement."""
    d = float(_axis_from_means(pos_mean, neg_mean).norm())
    if d <= 1e-12:
        raise ValueError("both prompts produced the same activations, so there is no direction to save. "
                         "Make the two ideas differ more (change the words that carry the idea).")
    mu = 0.5 * (float(pos_mean.norm()) + float(neg_mean.norm()))
    return round(min(SAFE_MAX_CEILING, fraction * mu / d), 4) or 0.0001


def _record_mean(model, block, module, model_name, cond_only, drive):
    """Run `drive(patched_model)` once with a read-only collect hook at `block`; return the [dim] mean.
    Same hook + wrapper machinery as HelmCollect, so the numbers are the ones Collect would write."""
    mp = model.clone()
    acc = {"sum": None, "n": 0, "last": False}
    fn = _make_collect_hook(acc, "(make-axis)", block, module, model_name, cond_only, "none")
    _add_hook(mp, {"block": int(block), "fn": fn, "cond_only": bool(cond_only), "start": 0.0, "end": 1.0})
    _install_wrapper(mp)
    drive(mp)
    if acc["sum"] is None or acc["n"] <= 0:
        raise RuntimeError(f"no activations were recorded at block {block}. The sampler never reached that "
                           f"block; check the block number and that the model is the one you sample with.")
    return acc["sum"] / acc["n"]


def _sample_for_axis(mp, clip, text, latent, steps, seed, cfg):
    """The real driver: encode `text` and run a short sample through ComfyUI's own KSampler path. Imported
    lazily because `nodes` is ComfyUI's core module, which exists only inside a running ComfyUI."""
    import nodes as core
    enc = core.CLIPTextEncode()
    positive = enc.encode(clip, text)[0]
    negative = enc.encode(clip, "")[0]
    core.common_ksampler(mp, int(seed), int(steps), float(cfg), "euler", "simple", positive, negative,
                         latent, denoise=1.0)


# --- nodes ---------------------------------------------------------------------------------
class HelmCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model to record from. The model passes through unchanged."}),
            "tag": ("STRING", {"default": "pos",
                               "tooltip": "Name of the recording. Record once with a prompt that has the idea and once without, with different names."}),
            "block": ("INT", {"default": 20, "min": 0, "max": 512, "tooltip": "Transformer block index to read."}),
            "cond_only": ("BOOLEAN", {"default": False,
                                      "tooltip": "Default False preserves the published result. True (recommended for CFG models) averages only cond batch items; current axes were NOT measured with cond_only."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm/Lab"
    DESCRIPTION = "Lab: record what one transformer block does while a prompt renders. Step 1 of the manual route to an axis."
    SEARCH_ALIASES = ["record", "collect activations", "advanced"]

    def run(self, model, tag, block, cond_only=False):
        _axes()
        mp = model.clone()
        entry = _calib_for(mp)
        module = entry.get("module") if entry else None
        model_name = _model_name(mp)
        acc = {"sum": None, "n": 0, "last": False}
        fn = _make_collect_hook(acc, tag, block, module, model_name, cond_only, "last")
        _add_hook(mp, {"block": int(block), "fn": fn, "cond_only": bool(cond_only),
                       "start": 0.0, "end": 1.0})
        _install_wrapper(mp)
        cal = f" [calibrated module={module}]" if entry else ""
        print(f"[helm] collecting '{tag}' at block {block} -> {_axis_path(tag)}{cal}")
        return (mp,)


class HelmVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pos_tag": ("STRING", {"default": "pos", "tooltip": "Name of the recording made WITH the idea."}),
            "neg_tag": ("STRING", {"default": "neg", "tooltip": "Name of the recording made WITHOUT the idea."}),
            "axis": ("STRING", {"default": "axis", "tooltip": "Name of the axis file to write."}),
            "safe_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "optional: the highest strength verified safe for this axis. 0 = unset. Stored with the axis and enforced by HelmApply."}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "run"
    CATEGORY = "Helm/Lab"
    DESCRIPTION = "Lab: subtract two recordings into one axis file. Step 2 of the manual route to an axis."
    SEARCH_ALIASES = ["build axis", "make vector", "advanced"]

    def run(self, pos_tag, neg_tag, axis, safe_max=0.0):
        p, n = _resolve_axis(pos_tag), _resolve_axis(neg_tag)
        v = _load_axis(p) - _load_axis(n)
        meta = _read_meta(pos_tag) or {}
        nmeta = _read_meta(neg_tag) or {}
        for k in ("block", "model"):
            if meta.get(k) is not None and nmeta.get(k) is not None and meta[k] != nmeta[k]:
                raise ValueError(
                    f"pos '{pos_tag}' has {k}={meta[k]!r} but neg '{neg_tag}' has {nmeta[k]!r}; "
                    f"an axis between different {k}s is meaningless.")
        _save_axis(axis, v, {"kind": "axis", "dim": int(v.shape[0]), "block": meta.get("block"),
                             "model": meta.get("model"), "module": meta.get("module"),
                             "norm": round(float(v.norm()), 3),
                             "safe_max": float(safe_max) if safe_max else None,
                             "from": {"pos": pos_tag, "neg": neg_tag}})
        print(f"[helm] axis '{axis}' dim={tuple(v.shape)} norm={float(v.norm()):.3f} safe_max={safe_max or '-'} -> {_axis_path(axis)}")
        return (axis,)


class HelmApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model to steer."}),
            "axis": ("STRING", {"default": "axis", "tooltip": "Name of the axis file to apply."}),
            "block": ("INT", {"default": 20, "min": -1, "max": 512,
                              "tooltip": "-1 = adopt the block recorded inside the axis file (the block it was built at); otherwise this explicit index"}),
            "strength": ("FLOAT", {"default": 0.1, "min": -10.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "scale of the injected axis. Model-specific; see calibration.json"}),
            "dial": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "optional normalized 0-1 dial. If >0 and the axis recorded a safe_max, strength = dial * safe_max (keeps you inside the verified band)."}),
            "cond_only": ("BOOLEAN", {"default": False,
                                      "tooltip": "Default False preserves the published result. True (recommended for CFG models) adds the axis only to cond batch items; current axes were NOT measured with cond_only."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "sampling-fraction window start. Default 0.0 preserves the published result."}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "sampling-fraction window end. Default 1.0 preserves the published result."}),
        }, "optional": {
            "mode": (["push", "norm", "clamp"], {"default": "push",
                     "tooltip": "push = published x+strength*axis. norm = same, rescaled to the token's original norm. clamp = move the axis projection, clamped to +/- clamp_k*||x||."}),
            "mask": ("STRING", {"default": "",
                     "tooltip": "path to a black/white PNG. White = apply the axis there (requires layout + grid). Empty = global (published behavior)."}),
            "layout": (["none", "h_w", "t_h_w", "h_w_t"], {"default": "none",
                     "tooltip": "token order of the block output. h_w = image; t_h_w = frame-major video; h_w_t = frame-inner. Never guessed: the mask is refused if the grid does not multiply to the token count."}),
            "grid": ("STRING", {"default": "",
                     "tooltip": "explicit H,W (image) or H,W,T (video). Required whenever a mask is given; the grid is never inferred."}),
            "clamp_k": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                     "tooltip": "clamp mode only: projection limit as a multiple of each token's norm."}),
            "mask_offset": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1,
                     "tooltip": "leading packed tokens to skip before the mask. Z-Image packs 32 text/cond rows before the 4096 spatial tokens; H3 packs [text|cond|audio|video], so pass the row count before the video slice. Requires mask."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm/Lab"
    DESCRIPTION = "Lab: apply an axis with full manual control (strength, block, window, mask). Most people want Helm · Dial instead."
    SEARCH_ALIASES = ["apply axis", "manual strength", "mask", "advanced"]

    def run(self, model, axis, block, strength, dial=0.0, cond_only=False, start_percent=0.0, end_percent=1.0,
            mode="push", mask="", layout="none", grid="", clamp_k=1.0, mask_offset=0):
        path = _resolve_axis(axis)
        vec = _load_axis(path)
        meta = _read_meta(axis) or {}
        try:
            meta_path = _resolve_meta(axis)
        except FileNotFoundError:
            meta_path = _meta_path(axis)

        if int(block) < 0:
            auto = meta.get("block")
            if auto is None:
                raise ValueError(
                    f"block=-1 (auto) but axis '{axis}' records no block index. Pass an explicit block, "
                    f"or rebuild the axis so its metadata carries one.")
            block = int(auto)

        safe = meta.get("safe_max")
        safe = float(safe) if safe else None
        if not safe:
            if dial:
                print(f"[helm] dial={dial} ignored: axis '{axis}' has no safe_max. "
                      f"Use strength directly, or rebuild the axis with a safe_max.")
            elif strength:
                print(f"[helm] warning: axis '{axis}' has no safe_max; strength={strength} is unclamped. "
                      f"Rebuild the axis with a safe_max to enable the verified band.")
        else:
            if dial:
                strength = round(dial * safe, 4)
            if abs(strength) > safe:
                print(f"[helm] strength {strength} exceeds the verified safe_max {safe} for '{axis}' -> clamped to {safe}. "
                      f"Past the band the frame washes out.")
                strength = safe if strength > 0 else -safe

        token_mask = None
        if mask:
            if layout == "none":
                raise ValueError("mask given without a layout; pass layout=h_w (image) or t_h_w/h_w_t (video)")
            if not grid:
                raise ValueError("mask given without grid; pass grid=H,W (image) or H,W,T (video) — Helm does not guess the token layout")
            gh, gw, gf = _parse_grid(grid)
            token_mask = (_load_mask(mask, gh, gw), layout, gf)
            print(f"[helm] mask={mask} layout={layout} grid={gh}x{gw}x{gf} mask_offset={int(mask_offset)}")
        elif int(mask_offset):
            raise ValueError("mask_offset given without mask")

        mp = model.clone()
        entry = _calib_for(mp)
        if entry and entry.get("dim") and int(entry["dim"]) != int(vec.shape[-1]):
            raise ValueError(
                f"axis '{axis}' dim {int(vec.shape[-1])} != calibrated model dim {int(entry['dim'])} "
                f"for {_model_name(mp)}; rebuild the axis on this model or pick a compatible one.")
        fn = _make_apply_hook(vec, strength, axis, cond_only, meta_path,
                              mode=mode, token_mask=token_mask, clamp_k=clamp_k, debug=True,
                              mask_offset=int(mask_offset))
        _add_hook(mp, {"block": int(block), "fn": fn, "cond_only": bool(cond_only),
                       "start": float(start_percent), "end": float(end_percent)})
        _install_wrapper(mp)
        cal = f", calibrated module={entry.get('module')}" if entry else ""
        print(f"[helm] applying '{axis}' at block {block}, strength={strength}, mode={mode} "
              f"(norm={float(vec.norm()):.2f}, safe_max={safe or '-'}, dim={vec.shape[-1]}, "
              f"window={start_percent}-{end_percent}{cal})")
        return (mp,)


class HelmDial:
    """The everyday control: pick an idea by name, move Amount. Delegates to HelmApply (same hook)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model to steer. Connect it between your model loader and the sampler."}),
            "axis": (_dial_choices(), {"default": _dial_default(_dial_choices()),
                                       "tooltip": "The idea this dial moves (for example 'Less clutter'). "
                                       "'Essentials' are ideas Helm's own benchmark shows still look right when moved, "
                                       "listed by how visible the change is. 'Subtle' ideas hold up too but change "
                                       "the picture only a little. 'Custom' are axes you made or dropped in, not "
                                       "benchmarked. The model name in brackets must match the model you loaded."}),
            "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                                 "tooltip": "How much of the idea. 0 = off (your image is untouched). 1 = the strongest "
                                            "setting that is still safe for this axis. Start in the middle."}),
        }, "optional": {
            "axis_from": ("STRING", {"default": "", "forceInput": True,
                                     "tooltip": "Optional. Connect the 'axis' output of Helm · Make Axis here to dial the axis you "
                                                "just made without picking it from the menu."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The same model with one idea dialed in. Connect it to your sampler.",)
    FUNCTION = "run"
    CATEGORY = "Helm"
    DESCRIPTION = ("Move one idea (light, clutter, motion...) with one slider, without rewriting the prompt. "
                   "Pick the idea, set Amount (0 = off, 1 = the strongest setting that still looks right), then Run.")
    SEARCH_ALIASES = ["helm", "dial", "slider", "one idea", "steer", "amount", "axis", "light", "clutter"]

    @classmethod
    def VALIDATE_INPUTS(cls, axis):
        # An axis made by Make Axis in this very run does not exist when the prompt is validated, so the menu
        # list cannot be checked up front. Existence is checked in run() with a readable message instead.
        return True if str(axis or "").strip() else "choose an axis"

    def run(self, model, axis, amount, axis_from=""):
        entry = _dial_resolve(axis_from if str(axis_from or "").strip() else axis)
        name = entry["file"]
        vec = _load_axis(_resolve_axis(name))
        meta = _read_meta(name) or {}
        _check_axis_fits_model(model, entry, meta, vec)
        safe = meta.get("safe_max") or entry.get("safe_max")
        strength = _dial_strength(amount, safe)
        block = meta.get("block")
        if block is None:
            calib = _calib_for(model)
            block = calib.get("default_block") if calib else None
        if block is None:
            raise ValueError(f"'{entry['label']}' does not record which block it was made at. Rebuild it "
                             f"with 'Helm · Make Axis'.")
        if str(meta.get("safe_max_source") or "").startswith("estimate"):
            print(f"[helm] '{entry['label']}' was made by Make Axis: its safe range is an estimate, not a "
                  f"measurement. Compare Before | After.")
        print(f"[helm] Dial: {_dial_words(entry['label'], amount)} "
              f"(strength {strength:.4f} of a safe range 0..{float(safe):g})")
        return HelmApply().run(model, name, int(block), strength)


class HelmMakeAxis:
    """Two prompts in, one axis file out. Records both contrast passes inside this single execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model you will render with, after any sampling-shift node."}),
            "clip": ("CLIP", {"tooltip": "The text encoder of the same model."}),
            "latent": ("LATENT", {"tooltip": "An empty latent. Its size is used for the two short test renders; "
                                  "a small size (512) is enough and faster."}),
            "with_idea": ("STRING", {"multiline": True, "default": "a quiet street at dawn, thick fog, documentary photography",
                                     "tooltip": "A prompt that HAS the idea. Keep everything else the same as the other prompt."}),
            "without_idea": ("STRING", {"multiline": True, "default": "a quiet street at dawn, clear air, documentary photography",
                                        "tooltip": "The same prompt WITHOUT the idea. Only the idea should differ."}),
            "name": ("STRING", {"default": "my_axis",
                                "tooltip": "Name of the axis. It is saved in your user/steer folder and then appears in the Dial menu under Custom."}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 30,
                              "tooltip": "Sampling steps for each of the two test renders. A few are enough; the images are thrown away."}),
        }, "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "tooltip": "Same seed for both test renders so only the prompt differs."}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                              "tooltip": "Guidance for the test renders. Use the value your model normally uses (1.0 for turbo models)."}),
            "block": ("INT", {"default": -1, "min": -1, "max": 512,
                              "tooltip": "-1 = the recommended block for this model (calibration.json), or the middle of the model if it is not calibrated."}),
        }}
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "axis")
    OUTPUT_TOOLTIPS = ("The model, unchanged. Connect it to a Helm · Dial to use the new axis straight away.",
                       "The name of the saved axis. Connect it to the 'axis_from' input of a Helm · Dial.")
    FUNCTION = "run"
    CATEGORY = "Helm"
    OUTPUT_NODE = True
    DESCRIPTION = ("Make your own idea for the Dial: write one prompt that has the idea and one that does not, "
                   "then Run. The idea is saved and appears in the Dial menu under Custom.")
    SEARCH_ALIASES = ["helm", "make axis", "new idea", "create axis", "my own axis", "custom idea"]

    def run(self, model, clip, latent, with_idea, without_idea, name, steps=4, seed=0, cfg=1.0, block=-1,
            _drive=None):
        axis_name = _safe_axis_name(name)
        entry = _calib_for(model)
        module = entry.get("module") if entry else None
        blocks, module = _find_blocks(model, which=module)
        if int(block) < 0:
            block = int(entry["default_block"]) if entry and entry.get("default_block") is not None else len(blocks) // 2
        if int(block) >= len(blocks):
            raise ValueError(f"block {int(block)} does not exist: this model has {len(blocks)} blocks (0-{len(blocks) - 1}).")
        model_name = _model_name(model)
        drive = _drive or (lambda mp, text: _sample_for_axis(mp, clip, text, latent, steps, seed, cfg))
        pos = _record_mean(model, block, module, model_name, False, lambda mp: drive(mp, with_idea))
        neg = _record_mean(model, block, module, model_name, False, lambda mp: drive(mp, without_idea))
        vec = _axis_from_means(pos, neg)
        safe = _estimate_safe_max(pos, neg)
        _save_axis(axis_name, vec, {
            "kind": "axis", "dim": int(vec.shape[0]), "block": int(block), "model": model_name, "module": module,
            "norm": round(float(vec.norm()), 3), "safe_max": safe,
            "safe_max_source": "estimate: 0.5% of the activation norm, not verified by renders",
            "from": {"pos": with_idea, "neg": without_idea}, "label": str(name).strip(),
            "made_by": "HelmMakeAxis"})
        print(f"[helm] Make Axis: saved '{axis_name}' (block {int(block)}, dim {vec.shape[0]}) -> "
              f"{_axis_path(axis_name)}. Its safe range 0..{safe} is an ESTIMATE; compare Before | After.")
        return (model, axis_name)


class HelmProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model to probe. The model passes through unchanged."}),
            "tag": ("STRING", {"default": "probe_pos",
                               "tooltip": "file to write. Run this node TWICE against the same model — once with a `pos` tag, once with a `neg` tag — then feed both to HelmProbeReport."}),
            "cond_only": ("BOOLEAN", {"default": False,
                                      "tooltip": "Default False preserves the published result. True averages only cond batch items (recommended for CFG models)."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm/Lab"
    DESCRIPTION = "Lab: record every block at once, to find where two prompts differ most. Run twice, then use Probe report."
    SEARCH_ALIASES = ["probe", "find block", "advanced"]

    def run(self, model, tag="probe_pos", cond_only=False):
        _axes()
        mp = model.clone()
        entry = _calib_for(mp)
        module = entry.get("module") if entry else None
        model_name = _model_name(mp)
        blocks, _ = _find_blocks(mp, which=module)
        n = len(blocks)
        acc = {"rows": None, "steps": 0, "n_blocks": n, "last": False}
        acc["reset"] = lambda: acc.update(rows=None, steps=0)
        fn = _make_probe_hook(acc, tag, module, model_name, cond_only, "last")
        _add_hook(mp, {"fn": fn, "all_blocks": True, "start": 0.0, "end": 1.0})
        _install_wrapper(mp)
        print(f"[helm] probing all {n} blocks under tag '{tag}' -> {_axis_path(tag)} "
              f"(run twice with different tags, then HelmProbeReport)")
        return (mp,)


class HelmProbeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pos_tag": ("STRING", {"default": "probe_pos", "tooltip": "Name of the probe made WITH the idea."}),
            "neg_tag": ("STRING", {"default": "probe_neg", "tooltip": "Name of the probe made WITHOUT the idea."}),
            "top": ("INT", {"default": 5, "min": 1, "max": 64,
                            "tooltip": "how many ranked blocks to print"}),
        }}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("peak_block",)
    FUNCTION = "run"
    CATEGORY = "Helm/Lab"
    DESCRIPTION = "Lab: rank the blocks of two probes by how much the prompts differ. A measurement, not a verified pick."
    SEARCH_ALIASES = ["probe report", "rank blocks", "advanced"]

    def run(self, pos_tag="probe_pos", neg_tag="probe_neg", top=5):
        P = _load_tensor(_resolve_axis(pos_tag))
        N = _load_tensor(_resolve_axis(neg_tag))
        if P.dim() != 2 or P.shape != N.shape:
            raise ValueError(
                f"probe files must both be [n_blocks, dim] tensors from HelmProbe with matching shapes; "
                f"got {tuple(P.shape)} for '{pos_tag}' and {tuple(N.shape)} for '{neg_tag}'")
        prof = _probe_profile(P, N)
        peak = prof["peak"]
        meta = _read_meta(pos_tag) or {}
        report = {"kind": "probe_report", "created": time.time(), "pos_tag": pos_tag, "neg_tag": neg_tag,
                  "model": meta.get("model"), "module": meta.get("module"), **prof}
        path = os.path.join(_axes(), "probe_report.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"[helm] could not write {path}: {e}")
        print("[helm] probe report — separation (high = the two prompts differ more at that block):")
        for row in prof["ranking"][:max(1, int(top))]:
            print(f"[helm]   block {row['block']:>3}: sep={row['separation']:.4f} cos={row['cosine']:+.4f} "
                  f"|dmu|={row['delta_norm']:.1f} |mu|={row['mu_norm']:.1f}")
        print(f"[helm] highest separation in the middle 50%: block {peak['block']} of {prof['n_blocks']} "
              f"(sep={peak['separation']}). This is a MEASUREMENT, not a validated block picker — on "
              f"Z-Image the peak sat at 7 while the band verified by rendering is 9-21. Sweep strength by "
              f"hand: the scale is model-specific. Report -> {path}")
        return (int(peak["block"]),)


NODE_CLASS_MAPPINGS = {"HelmDial": HelmDial, "HelmMakeAxis": HelmMakeAxis,
                       "HelmCollect": HelmCollect, "HelmVector": HelmVector, "HelmApply": HelmApply,
                       "HelmProbe": HelmProbe, "HelmProbeReport": HelmProbeReport}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HelmDial": "Helm \u00b7 Dial",
    "HelmMakeAxis": "Helm \u00b7 Make Axis",
    "HelmCollect": "Helm Lab \u00b7 Record activations",
    "HelmVector": "Helm Lab \u00b7 Build axis from two recordings",
    "HelmApply": "Helm Lab \u00b7 Apply axis (manual strength)",
    "HelmProbe": "Helm Lab \u00b7 Probe every block",
    "HelmProbeReport": "Helm Lab \u00b7 Probe report",
}
