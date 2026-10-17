"""Helm — training-free activation steering for ComfyUI.

Read a transformer block's mean activation for two contrastive prompts (diff-in-means) and inject
`x + strength*axis` at that block during sampling. Weights are not modified; an axis is a small
`.safetensors` file that carries its own metadata (dim, block, model). Reads resolve `user/steer/` ->
`models/steer/` -> this pack's `example_axes/`; writes always go to `user/steer/`. Axes written by
older versions as `.pt` are still read; `HelmApply` `block=-1` adopts the block recorded in the axis.

Nodes
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
    try:
        with open(_resolve_meta(name), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    try:
        return _load_axis_meta(_resolve_axis(name))
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
def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _stack_mean(o, flags=None, cond_only=False):
    x = _first(o).detach().float()
    if cond_only and flags is not None:
        try:
            f = torch.as_tensor(flags).reshape(-1)
            if f.numel() == x.shape[0] and f.numel() > 0:
                sel = x[f.to(x.device) == 0]
                if sel.shape[0] > 0:
                    x = sel
        except Exception:
            pass
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


_BLOCK_ATTRS = ["blocks", "layers", "transformer_blocks", "double_blocks", "single_blocks"]


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
        v = _stack_mean(out, flags=flags, cond_only=cond_only)
        acc["sum"] = v if acc["sum"] is None else acc["sum"] + v
        acc["n"] += 1
        if write_mode == "each" or (write_mode == "last" and acc.get("last")):
            _flush(tag, acc, block, module, model_name)
        return None  # read-only
    fn._acc = acc
    return fn


def _make_apply_hook(vec, strength, axis_name, cond_only, meta_path):
    def fn(m, args, out, flags=None):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[-1] != vec.shape[-1]:
            raise RuntimeError(
                f"axis '{axis_name}' dim {vec.shape[-1]} != block output dim {x.shape[-1]}; "
                f"this axis was built for a different model (see {meta_path})")
        d = vec.to(x.dtype).to(x.device) * strength
        if cond_only and flags is not None:
            d = _mask_by_flags(d, x, flags)
        y = x + d
        return (y,) + out[1:] if isinstance(out, tuple) else y
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
        v = _stack_mean(out, flags=flags, cond_only=cond_only)
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


# --- nodes ---------------------------------------------------------------------------------
class HelmCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tag": ("STRING", {"default": "pos"}),
            "block": ("INT", {"default": 20, "min": 0, "max": 512, "tooltip": "transformer block index to read"}),
            "cond_only": ("BOOLEAN", {"default": False,
                                      "tooltip": "Default False preserves the published result. True (recommended for CFG models) averages only cond batch items; current axes were NOT measured with cond_only."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm"

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
            "pos_tag": ("STRING", {"default": "pos"}),
            "neg_tag": ("STRING", {"default": "neg"}),
            "axis": ("STRING", {"default": "axis"}),
            "safe_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "optional: the highest strength verified safe for this axis. 0 = unset. Stored with the axis and enforced by HelmApply."}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "run"
    CATEGORY = "Helm"

    def run(self, pos_tag, neg_tag, axis, safe_max=0.0):
        p, n = _resolve_axis(pos_tag), _resolve_axis(neg_tag)
        v = _load_axis(p) - _load_axis(n)
        meta = _read_meta(pos_tag) or {}
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
            "model": ("MODEL",),
            "axis": ("STRING", {"default": "axis"}),
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
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm"

    def run(self, model, axis, block, strength, dial=0.0, cond_only=False, start_percent=0.0, end_percent=1.0):
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
        if dial and safe:
            strength = round(dial * float(safe), 4)
        if safe and abs(strength) > float(safe):
            print(f"[helm] strength {strength} exceeds the verified safe_max {safe} for '{axis}' -> clamped to {safe}. "
                  f"Past the band the frame washes out.")
            strength = float(safe) if strength > 0 else -float(safe)

        mp = model.clone()
        entry = _calib_for(mp)
        fn = _make_apply_hook(vec, strength, axis, cond_only, meta_path)
        _add_hook(mp, {"block": int(block), "fn": fn, "cond_only": bool(cond_only),
                       "start": float(start_percent), "end": float(end_percent)})
        _install_wrapper(mp)
        cal = f", calibrated module={entry.get('module')}" if entry else ""
        print(f"[helm] applying '{axis}' at block {block}, strength={strength} "
              f"(norm={float(vec.norm()):.2f}, safe_max={safe or '-'}, dim={vec.shape[-1]}, "
              f"window={start_percent}-{end_percent}{cal})")
        return (mp,)


class HelmProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tag": ("STRING", {"default": "probe_pos",
                               "tooltip": "file to write. Run this node TWICE against the same model — once with a `pos` tag, once with a `neg` tag — then feed both to HelmProbeReport."}),
            "cond_only": ("BOOLEAN", {"default": False,
                                      "tooltip": "Default False preserves the published result. True averages only cond batch items (recommended for CFG models)."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm"

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
            "pos_tag": ("STRING", {"default": "probe_pos"}),
            "neg_tag": ("STRING", {"default": "probe_neg"}),
            "top": ("INT", {"default": 5, "min": 1, "max": 64,
                            "tooltip": "how many ranked blocks to print"}),
        }}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("peak_block",)
    FUNCTION = "run"
    CATEGORY = "Helm"

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


NODE_CLASS_MAPPINGS = {"HelmCollect": HelmCollect, "HelmVector": HelmVector, "HelmApply": HelmApply,
                       "HelmProbe": HelmProbe, "HelmProbeReport": HelmProbeReport}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HelmCollect": "Helm Collect (read activation)",
    "HelmVector": "Helm Axis (diff-in-means)",
    "HelmApply": "Helm Apply (strength x axis)",
    "HelmProbe": "Helm Probe (read all blocks)",
    "HelmProbeReport": "Helm Probe Report (pick block + strength)",
}
