"""Helm — training-free activation steering for ComfyUI.

Read a transformer block's mean activation for two contrastive prompts (diff-in-means) and inject
`x + strength*axis` at that block during sampling. Weights are not modified; an axis is a small tensor
file under models/steer/.

Nodes
  HelmCollect  MODEL -> MODEL   mean activation at block K for a tag (run once per prompt)
  HelmVector   tags  -> STRING  axis = mean_pos - mean_neg -> reusable axis file
  HelmApply    MODEL -> MODEL   inject x + strength*axis at block K (chain several for multiple axes)

Wire MODEL source -> HelmApply -> sampler. Works on any DiT exposing a `blocks`/`layers` ModuleList
(H3: blocks, dim 5376, block ~20; Z-Image/Lumina: layers, dim 3840, layers 9-21). Per-model defaults are
in calibration.json; the axis scale is model-specific.
"""
import json
import os
import time

import torch
import folder_paths

# --- paths / calibration -------------------------------------------------------------------
AXIS_DIR = os.path.join(folder_paths.get_user_directory(), "steer")
_CALIB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration.json")


def _axes():
    os.makedirs(AXIS_DIR, exist_ok=True)
    return AXIS_DIR


def _axis_path(name):
    return os.path.join(_axes(), f"{name}.pt")


def _meta_path(name):
    return os.path.join(_axes(), f"{name}.json")


def _write_meta(name, meta):
    try:
        meta["helm_version"] = 1
        meta["created"] = None  # set by caller
        with open(_meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[helm] could not write metadata for '{name}': {e}")


def _read_meta(name):
    try:
        with open(_meta_path(name), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _calibration():
    try:
        with open(_CALIB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"models": []}


# --- hook machinery ------------------------------------------------------------------------
def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _stack_mean(o):
    x = _first(o).detach().float()
    while x.dim() > 1:
        x = x.mean(dim=0)
    return x  # -> [dim]


def _find_blocks(model, which=None):
    dm = model.model.diffusion_model
    names = [which] if which else ["blocks", "layers"]
    for n in names:
        mod = getattr(dm, n, None)
        if mod is not None and len(mod) > 4:
            return getattr(dm, n)
    raise RuntimeError(
        "no transformer block list on diffusion_model (looked for 'blocks'/'layers'); "
        "pass the ModuleList name via the overrides in calibration.json or open an issue"
    )


def _wrapper(mp, block_idx, on_step):
    """ComfyUI calls `model_function_wrapper(apply_model, {"input","timestep","c","cond_or_uncond"})`.
    Install the forward hook only for the model call (one step), then remove it. Aimdo (H3's per-block
    prefetch / comfy_aimdo malloc graph) raises 'aimdo memory compile error' when a block's output is
    modified, so disable it for the call and restore it after."""
    def run(apply_model, c_in):
        import comfy.memory_management as cm
        prev = getattr(cm, "aimdo_enabled", None)
        if prev is not None:
            cm.aimdo_enabled = False
        blocks = _find_blocks(mp)
        if block_idx >= len(blocks):
            raise RuntimeError(f"block {block_idx} >= {len(blocks)} on {mp.model.diffusion_model.__class__.__name__}")
        handle = blocks[block_idx].register_forward_hook(on_step)
        try:
            return apply_model(c_in["input"], c_in["timestep"], **c_in["c"], cond_or_uncond=c_in["cond_or_uncond"])
        finally:
            handle.remove()
            if prev is not None:
                cm.aimdo_enabled = prev
    return run


# --- nodes ---------------------------------------------------------------------------------
class HelmCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tag": ("STRING", {"default": "pos"}),
            "block": ("INT", {"default": 20, "min": 0, "max": 512, "tooltip": "transformer block index to read"}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm"

    def run(self, model, tag, block):
        _axes()
        mp = model.clone()
        acc = {"sum": None, "n": 0}

        def hook(m, i, o):
            v = _stack_mean(o)
            acc["sum"] = v if acc["sum"] is None else acc["sum"] + v
            acc["n"] += 1
            torch.save(acc["sum"] / max(acc["n"], 1), _axis_path(tag))
            _write_meta(tag, {"kind": "activation", "dim": int(v.shape[0]), "block": int(block),
                              "created": time.time()})
            return None  # read-only

        mp.set_model_unet_function_wrapper(_wrapper(mp, block, hook))
        print(f"[helm] collecting '{tag}' at block {block} -> {_axis_path(tag)}")
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
        p, n = _axis_path(pos_tag), _axis_path(neg_tag)
        if not (os.path.exists(p) and os.path.exists(n)):
            raise FileNotFoundError(f"missing axis inputs: {p if not os.path.exists(p) else n} (run HelmCollect for both prompts first)")
        v = torch.load(p, map_location="cpu") - torch.load(n, map_location="cpu")
        meta = _read_meta(pos_tag) or {}
        torch.save(v, _axis_path(axis))
        _write_meta(axis, {"kind": "axis", "dim": int(v.shape[0]), "block": meta.get("block"),
                           "norm": round(float(v.norm()), 3),
                           "safe_max": float(safe_max) if safe_max else None,
                           "from": {"pos": pos_tag, "neg": neg_tag}, "created": time.time()})
        print(f"[helm] axis '{axis}' dim={tuple(v.shape)} norm={float(v.norm()):.3f} safe_max={safe_max or '-'} -> {_axis_path(axis)}")
        return (axis,)


class HelmApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "axis": ("STRING", {"default": "axis"}),
            "block": ("INT", {"default": 20, "min": 0, "max": 512}),
            "strength": ("FLOAT", {"default": 0.1, "min": -10.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "scale of the injected axis. Model-specific; see calibration.json"}),
            "dial": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "optional normalized 0-1 dial. If >0 and the axis recorded a safe_max, strength = dial * safe_max (keeps you inside the verified band)."}),
        }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Helm"

    def run(self, model, axis, block, strength, dial=0.0):
        path = _axis_path(axis)
        if not os.path.exists(path):
            raise FileNotFoundError(f"axis '{axis}' not found at {path} (run HelmCollect + HelmVector first)")
        vec = torch.load(path, map_location="cpu")
        meta = _read_meta(axis) or {}

        safe = meta.get("safe_max")
        if dial and safe:
            strength = round(dial * float(safe), 4)
        if safe and abs(strength) > float(safe):
            print(f"[helm] strength {strength} exceeds the verified safe_max {safe} for '{axis}' -> clamped to {safe}. "
                  f"Past the band the frame washes out.")
            strength = float(safe) if strength > 0 else -float(safe)

        mp = model.clone()

        def hook(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            if x.shape[-1] != vec.shape[-1]:
                raise RuntimeError(
                    f"axis '{axis}' dim {vec.shape[-1]} != block output dim {x.shape[-1]}; "
                    f"this axis was built for a different model (see {_meta_path(axis)})")
            d = vec.to(x.dtype).to(x.device)
            if isinstance(o, tuple):
                return (o[0] + strength * d,) + o[1:]
            return o + strength * d

        mp.set_model_unet_function_wrapper(_wrapper(mp, block, hook))
        print(f"[helm] applying '{axis}' at block {block}, strength={strength} "
              f"(norm={float(vec.norm()):.2f}, safe_max={safe or '-'}, dim={vec.shape[-1]})")
        return (mp,)


NODE_CLASS_MAPPINGS = {"HelmCollect": HelmCollect, "HelmVector": HelmVector, "HelmApply": HelmApply}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HelmCollect": "Helm Collect (read activation)",
    "HelmVector": "Helm Axis (diff-in-means)",
    "HelmApply": "Helm Apply (strength x axis)",
}
