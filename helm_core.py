"""Helm core: the `HELM` data type and the pure maths behind the 0.5 node family.

Nothing here touches ComfyUI. The nodes in `nodes.py` call these functions; the tests exercise them on
synthetic tensors.

The HELM type is a small, immutable, validated object: a tuple of `HelmAxis` entries (vector + metadata +
amount, curve, stage, weight). Nodes never mutate one; they return a new one. It serializes to plain JSON
types (`to_dict` / `from_dict`) and is validated strictly on construction and on load: no pickle, no
unknown keys.

Methods implemented here, independently, from the published descriptions only:
  * Activation Addition (Turner et al., arXiv:2308.10248): steer by adding the difference of activations
    for contrasting prompts. Helm averages the difference over several prompt pairs.
  * Concept Sliders (arXiv:2311.12092): a signed, continuous control for one concept direction.
"""
import math
from dataclasses import dataclass, field, replace

import torch

HELM_VERSION = 1
MAX_AXES = 16
MAX_KEYFRAMES = 8
STAGES = ("all", "early", "late")
EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out")
CURVE_KINDS = ("constant", "ramp", "keys")
SOURCES = ("library", "pairs", "other")
_INFO_TYPES = (str, int, float, bool, type(None))


class HelmValueError(ValueError):
    """A HELM object (or one of its parts) failed validation."""


# --- validation helpers ----------------------------------------------------------------------
def _num(value, what, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HelmValueError(f"{what} must be a number (got {type(value).__name__})")
    v = float(value)
    if not math.isfinite(v):
        raise HelmValueError(f"{what} must be finite (got {value!r})")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise HelmValueError(f"{what} must be between {lo} and {hi} (got {v})")
    return v


def _text(value, what, max_len=200, allow_empty=False):
    if not isinstance(value, str):
        raise HelmValueError(f"{what} must be text (got {type(value).__name__})")
    if not allow_empty and not value.strip():
        raise HelmValueError(f"{what} must not be empty")
    if len(value) > max_len:
        raise HelmValueError(f"{what} is too long ({len(value)} > {max_len} characters)")
    return value


# --- the curve ---------------------------------------------------------------------------------
def ease(name, t):
    """Pure. Easing of t in [0, 1] (tensor or float) -> same shape in [0, 1]."""
    if name == "linear":
        return t
    if name == "ease_in":
        return t * t
    if name == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if name == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    raise HelmValueError(f"unknown easing {name!r}; expected one of {EASINGS}")


@dataclass(frozen=True, eq=False)
class HelmCurve:
    """How an axis amount changes along the clip's time axis (video only).

    constant: no change over time.
    ramp:     from the axis `amount` (at the first frame) to `end` (at the last), shaped by `easing`.
    keys:     piecewise-linear through `keys` = ((position 0..1, amount -1..1), ...); held flat outside.
    """
    kind: str = "constant"
    end: float = 0.0
    easing: str = "linear"
    keys: tuple = ()

    def __post_init__(self):
        if self.kind not in CURVE_KINDS:
            raise HelmValueError(f"curve kind must be one of {CURVE_KINDS} (got {self.kind!r})")
        object.__setattr__(self, "end", _num(self.end, "curve end", -1.0, 1.0))
        if self.easing not in EASINGS:
            raise HelmValueError(f"easing must be one of {EASINGS} (got {self.easing!r})")
        keys = tuple((float(t), float(a)) for t, a in self.keys)
        if self.kind == "keys":
            if not 2 <= len(keys) <= MAX_KEYFRAMES:
                raise HelmValueError(f"keyframes need 2 to {MAX_KEYFRAMES} entries (got {len(keys)})")
            prev = -1.0
            for t, a in keys:
                _num(t, "keyframe position", 0.0, 1.0)
                _num(a, "keyframe amount", -1.0, 1.0)
                if t <= prev:
                    raise HelmValueError("keyframe positions must be strictly increasing")
                prev = t
        elif keys:
            raise HelmValueError("keys are only allowed when the curve kind is 'keys'")
        object.__setattr__(self, "keys", keys)

    @property
    def is_constant(self):
        return self.kind == "constant"

    def to_dict(self):
        return {"kind": self.kind, "end": self.end, "easing": self.easing,
                "keys": [list(k) for k in self.keys]}

    @classmethod
    def from_dict(cls, d):
        _exact_keys(d, {"kind", "end", "easing", "keys"}, "curve")
        return cls(kind=d["kind"], end=d["end"], easing=d["easing"], keys=tuple(tuple(k) for k in d["keys"]))


def curve_series(base_amount, curve, ts):
    """Pure. The signed amount (-1..1) at each normalized time in `ts` (tensor in [0, 1])."""
    ts = torch.as_tensor(ts, dtype=torch.float32)
    base = _num(base_amount, "amount", -1.0, 1.0)
    if curve is None or curve.kind == "constant":
        return torch.full_like(ts, base)
    if curve.kind == "ramp":
        return base + (curve.end - base) * ease(curve.easing, ts.clamp(0.0, 1.0))
    kt = torch.tensor([k[0] for k in curve.keys], dtype=torch.float32)
    ka = torch.tensor([k[1] for k in curve.keys], dtype=torch.float32)
    out = torch.empty_like(ts)
    flat = ts.reshape(-1)
    res = torch.empty_like(flat)
    for i, t in enumerate(flat.tolist()):
        if t <= float(kt[0]):
            res[i] = ka[0]
        elif t >= float(kt[-1]):
            res[i] = ka[-1]
        else:
            j = int(torch.searchsorted(kt, torch.tensor(t), right=True)) - 1
            f = (t - float(kt[j])) / float(kt[j + 1] - kt[j])
            res[i] = ka[j] + (ka[j + 1] - ka[j]) * f
    return res.reshape(out.shape)


def frame_times(n_frames):
    """Pure. Normalized position of each of `n_frames` frames: 0 for the first, 1 for the last."""
    n = int(n_frames)
    if n < 1:
        raise HelmValueError("a clip needs at least one frame")
    if n == 1:
        return torch.zeros(1)
    return torch.arange(n, dtype=torch.float32) / (n - 1)


def parse_keyframes(text):
    """Pure. '0:0.2, 0.5:0.8, 1:0.4' -> ((0.0, 0.2), (0.5, 0.8), (1.0, 0.4)). Validated by HelmCurve."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise HelmValueError(f"keyframe {part!r} must look like position:amount, for example 0.5:0.8")
        t, a = part.split(":", 1)
        try:
            out.append((float(t), float(a)))
        except ValueError:
            raise HelmValueError(f"keyframe {part!r} must look like position:amount, for example 0.5:0.8")
    return tuple(out)


# --- stage (which part of the render) ---------------------------------------------------------
def progress_from_sigma(sigma, sample_sigmas, step=0, total=None):
    """Pure. How far through the render this call is, 0 (pure noise) .. 1 (finished).

    Uses the sigma ComfyUI passes to the model, normalized against the sampling schedule
    (`transformer_options['sample_sigmas']`), so a shifted schedule is respected. Falls back to the step
    counter when the schedule is missing or the sigma lies outside it (so a unit mismatch can never
    silently mis-gate a stage)."""
    if sigma is not None and sample_sigmas is not None:
        sig = [float(s) for s in sample_sigmas]
        if len(sig) >= 2 and sig[0] > sig[-1]:
            s = float(sigma)
            tol = 1e-3 * sig[0]
            if sig[-1] - tol <= s <= sig[0] + tol:
                return min(1.0, max(0.0, (sig[0] - s) / (sig[0] - sig[-1])))
    if total:
        return min(1.0, max(0.0, step / total))
    return 0.0


def stage_window(stage, split=0.5):
    """Pure. (lo, hi) progress window for a stage; active when lo <= progress < hi (hi=1.0 includes 1.0)."""
    if stage not in STAGES:
        raise HelmValueError(f"stage must be one of {STAGES} (got {stage!r})")
    split = _num(split, "split", 0.05, 0.95)
    return {"all": (0.0, 1.0), "early": (0.0, split), "late": (split, 1.0)}[stage]


def stage_active(window, progress):
    lo, hi = window
    return lo <= progress < hi or (hi >= 1.0 and progress >= 1.0 and lo < 1.0)


# --- H3 token layout -------------------------------------------------------------------------
def h3_video_layout(seq_len, latent_shape):
    """Pure. Where the video tokens sit in a MiniMax H3 block output.

    Derived from ComfyUI core 0.43.2, comfy/ldm/minimax/model.py: the packed sequence is
    [text | cond | ref | audio | video] with the target video ALWAYS the last segment, it
    holds `latent_t * frame_rows` rows (lines 443-455) and frame_rows is (H/2)*(W/2) of one padded latent frame
    (`_frame_grid`, lines 88-92). Rows are frame-major, then h, then w (`patchify_video`, lines 42-49;
    `_video_grid`, lines 126-130). The block output `h` is a 2-D [sequence, hidden] tensor (line 592).
    `latent_shape` is [B, C, T, H, W] (the first entry of the sampler's
    `latent_shapes`, model_base.py lines 2228-2230); H and W are padded up to a multiple of the 1x2x2 patch (line 652).

    Returns (first_video_row, n_frames, rows_per_frame). Raises HelmValueError when the shape cannot fit
    the block's sequence, so a wrong layout is refused rather than guessed."""
    shape = [int(v) for v in latent_shape]
    if len(shape) != 5:
        raise HelmValueError(f"expected a video latent shape [B, C, T, H, W] (got {shape})")
    t, h, w = shape[2], shape[3], shape[4]
    rows_per_frame = ((h + 1) // 2) * ((w + 1) // 2)
    n_video = t * rows_per_frame
    if t < 1 or rows_per_frame < 1 or n_video > int(seq_len):
        raise HelmValueError(f"the video latent {shape} needs {n_video} rows but the block has {seq_len}")
    return int(seq_len) - n_video, t, rows_per_frame


def frame_row_scale(seq_len, layout, frame_scale, outside=0.0):
    """Pure. Per-row multiplier [seq_len]: `outside` for non-video rows, then each frame's scale repeated over
    that frame's rows, in the layout order of `h3_video_layout`."""
    first, n_frames, rows_per_frame = layout
    fs = torch.as_tensor(frame_scale, dtype=torch.float32).reshape(-1)
    if fs.numel() != n_frames:
        raise HelmValueError(f"{fs.numel()} frame scales for {n_frames} frames")
    out = torch.full((int(seq_len),), float(outside), dtype=torch.float32)
    out[first:] = fs.repeat_interleave(rows_per_frame)
    return out


# --- maths on vectors ------------------------------------------------------------------------
def pair_diffs(pos_means, neg_means):
    """Pure. Stack the per-pair differences [n_pairs, dim] = with_idea - without_idea."""
    if len(pos_means) != len(neg_means) or not pos_means:
        raise HelmValueError("need the same, non-zero number of with-idea and without-idea recordings")
    rows = []
    for p, n in zip(pos_means, neg_means):
        p, n = p.to(torch.float32).reshape(-1), n.to(torch.float32).reshape(-1)
        if p.shape != n.shape:
            raise HelmValueError("a pair produced two different widths")
        rows.append(p - n)
    return torch.stack(rows)


def average_pairs(diffs):
    """Pure. The multipurpose axis: the mean of the per-pair differences. Subject-specific detail points in a
    different direction for every pair, so it shrinks by averaging; what the pairs share stays."""
    return diffs.mean(dim=0)


def pair_agreement(diffs):
    """Pure. Cosine of each pair's difference with the mean difference. Near 1 = the pairs agree on one
    direction; near 0 = the pairs share nothing (a sign the idea is not one direction)."""
    mean = average_pairs(diffs)
    mn = mean.norm()
    if float(mn) < 1e-12:
        return [0.0] * diffs.shape[0]
    return [float((d @ mean) / (d.norm() * mn + 1e-12)) for d in diffs]


def nuisance_basis(diffs, rel_tol=1e-3):
    """Pure. Orthonormal basis Q [r, dim] of the directions the nuisance pairs span.

    Each row of `diffs` is the mean-activation difference of one nuisance pair (same meaning, different
    wording; or same template, different subject; or just a different look). Rows are normalized so no single
    pair dominates, then an SVD keeps the directions with singular value above `rel_tol` of the largest."""
    rows = []
    for d in diffs:
        n = float(d.norm())
        if n > 1e-12:
            rows.append(d.to(torch.float32) / n)
    if not rows:
        raise HelmValueError("the nuisance pairs produced no direction at all")
    m = torch.stack(rows)
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    keep = int((s > rel_tol * s.max()).sum())
    return vh[:keep].contiguous()


def debias_project(vec, basis, strength=1.0):
    """Pure. Remove the nuisance subspace from an axis: v - strength * Q^T Q v.

    Returns (new_vec, removed_energy) where removed_energy in [0, 1] is the share of ||v||^2 that lay in
    the nuisance subspace (1 = the axis was entirely nuisance)."""
    s = _num(strength, "debias strength", 0.0, 1.0)
    v = vec.to(torch.float32).reshape(-1)
    if basis is None or basis.shape[0] == 0 or s == 0.0:
        return v.clone(), 0.0
    q = basis.to(torch.float32)
    comp = (v @ q.T) @ q
    out = v - s * comp
    n0 = float(v @ v)
    return out, (float(comp @ comp) / n0 if n0 > 0 else 0.0)


def gram_schmidt(vectors, preserve_norm=True, tol=1e-6):
    """Pure. Modified Gram-Schmidt in order: each vector loses its components along the earlier ones, so
    moving one afterwards leaves the others where they were. The first vector is never changed.

    With `preserve_norm` each result is rescaled to its original length, so an amount keeps its meaning.
    A vector that lies (almost) inside the span of the earlier ones has nothing left; it is returned as
    None and its index is listed in `dropped`. Returns (list of tensors or None, dropped indices)."""
    basis, out, dropped = [], [], []
    for i, v in enumerate(vectors):
        v = v.to(torch.float32).reshape(-1)
        n0 = float(v.norm())
        r = v.clone()
        for q in basis:
            r = r - (r @ q) * q
        for q in basis:  # second pass: re-orthogonalize against rounding
            r = r - (r @ q) * q
        nr = float(r.norm())
        if n0 <= 1e-12 or nr <= tol * max(n0, 1.0):
            out.append(None)
            dropped.append(i)
            continue
        basis.append(r / nr)
        out.append(r * (n0 / nr) if preserve_norm else r)
    return out, dropped


def parse_pairs(text, sep="||"):
    """Pure. Multi-line text, one pair per line 'with the idea || without the idea' -> [(with, without), ...]."""
    out = []
    for n, line in enumerate(str(text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if sep not in line:
            raise HelmValueError(f"line {n}: write each pair as 'with the idea {sep} without the idea'")
        a, b = (s.strip() for s in line.split(sep, 1))
        if not a or not b:
            raise HelmValueError(f"line {n}: both sides of the pair need text")
        if a == b:
            raise HelmValueError(f"line {n}: the two sides are identical, so there is no idea to extract")
        out.append((a, b))
    return out


def parse_weights(text, n):
    """Pure. '1, 0.5, 0.8' -> n floats in [0, 1]. Empty text -> all ones."""
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        return [1.0] * n
    if len(parts) != n:
        raise HelmValueError(f"axis_weights has {len(parts)} values but there are {n} axes")
    try:
        return [_num(float(p), "axis weight", 0.0, 1.0) for p in parts]
    except ValueError:
        raise HelmValueError("axis_weights must be numbers separated by commas")


# --- the HELM type ----------------------------------------------------------------------------
def _exact_keys(d, allowed, what):
    if not isinstance(d, dict):
        raise HelmValueError(f"{what} must be a mapping")
    extra = set(d) - allowed
    missing = allowed - set(d)
    if extra or missing:
        raise HelmValueError(f"{what}: unexpected keys {sorted(extra)}, missing keys {sorted(missing)}")


@dataclass(frozen=True, eq=False)
class HelmAxis:
    """One idea, how much, and when. Immutable; use `with_changes` for a modified copy."""
    name: str
    label: str
    vec: torch.Tensor
    model: str = None
    module: str = None
    block: int = None
    safe_max: float = None
    source: str = "other"
    amount: float = 0.5
    weight: float = 1.0
    curve: HelmCurve = None
    stage: str = "all"
    split: float = 0.5
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.name, "axis name", 120)
        _text(self.label, "axis label", 120)
        if not isinstance(self.vec, torch.Tensor):
            raise HelmValueError("axis vector must be a tensor")
        if self.vec.dim() != 1 or self.vec.numel() < 1:
            raise HelmValueError(f"axis vector must be 1-D (got shape {tuple(self.vec.shape)})")
        if not bool(torch.isfinite(self.vec).all()):
            raise HelmValueError(f"axis '{self.label}' contains NaN or infinite values")
        if float(self.vec.norm()) <= 1e-12:
            raise HelmValueError(f"axis '{self.label}' is all zeros, so it moves nothing")
        v = self.vec.detach().to(torch.float32).cpu().contiguous()
        object.__setattr__(self, "vec", v)
        for key in ("model", "module"):
            val = getattr(self, key)
            if val is not None:
                _text(val, key, 120)
        if self.block is not None:
            if isinstance(self.block, bool) or not isinstance(self.block, int) or self.block < 0:
                raise HelmValueError(f"block must be a whole number >= 0 (got {self.block!r})")
        if self.safe_max is not None:
            sm = _num(self.safe_max, "safe_max")
            if sm <= 0:
                raise HelmValueError(f"safe_max must be > 0 (got {sm})")
            object.__setattr__(self, "safe_max", sm)
        if self.source not in SOURCES:
            raise HelmValueError(f"source must be one of {SOURCES} (got {self.source!r})")
        object.__setattr__(self, "amount", _num(self.amount, "amount", -1.0, 1.0))
        object.__setattr__(self, "weight", _num(self.weight, "weight", 0.0, 1.0))
        if self.curve is not None and not isinstance(self.curve, HelmCurve):
            raise HelmValueError("curve must be a HelmCurve")
        if self.stage not in STAGES:
            raise HelmValueError(f"stage must be one of {STAGES} (got {self.stage!r})")
        object.__setattr__(self, "split", _num(self.split, "split", 0.05, 0.95))
        if not isinstance(self.info, dict) or len(self.info) > 16:
            raise HelmValueError("info must be a mapping of at most 16 entries")
        info = {}
        for k, val in self.info.items():
            _text(k, "info key", 60)
            if not isinstance(val, _INFO_TYPES):
                raise HelmValueError(f"info['{k}'] must be text, a number, a boolean or empty")
            if isinstance(val, str):
                _text(val, f"info['{k}']", 400, allow_empty=True)
            if isinstance(val, float) and not math.isfinite(val):
                raise HelmValueError(f"info['{k}'] must be finite")
            info[k] = val
        object.__setattr__(self, "info", info)

    @property
    def dim(self):
        return int(self.vec.shape[0])

    @property
    def has_curve(self):
        return self.curve is not None and not self.curve.is_constant

    def series(self, ts):
        """Signed amount (-1..1) at the normalized times `ts`, before the weight."""
        return curve_series(self.amount, self.curve, ts)

    def mean_amount(self):
        """Time-average of the curve; what a model without a known video layout falls back to."""
        return float(self.series(torch.linspace(0.0, 1.0, 101)).mean())

    def effective_amount(self):
        return self.mean_amount() * self.weight if self.has_curve else self.amount * self.weight

    @property
    def is_active(self):
        if self.weight == 0.0:
            return False
        ts = torch.linspace(0.0, 1.0, 101)
        return bool((self.series(ts).abs() > 1e-9).any())

    def with_changes(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return {"name": self.name, "label": self.label, "vec": self.vec.tolist(), "model": self.model,
                "module": self.module, "block": self.block, "safe_max": self.safe_max, "source": self.source,
                "amount": self.amount, "weight": self.weight,
                "curve": self.curve.to_dict() if self.curve else None,
                "stage": self.stage, "split": self.split, "info": dict(self.info)}

    @classmethod
    def from_dict(cls, d):
        _exact_keys(d, {"name", "label", "vec", "model", "module", "block", "safe_max", "source", "amount",
                        "weight", "curve", "stage", "split", "info"}, "axis")
        if not isinstance(d["vec"], list):
            raise HelmValueError("axis vec must be a list of numbers")
        for x in d["vec"]:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise HelmValueError("axis vec must contain only numbers")
        return cls(name=d["name"], label=d["label"], vec=torch.tensor(d["vec"], dtype=torch.float32),
                   model=d["model"], module=d["module"], block=d["block"], safe_max=d["safe_max"],
                   source=d["source"], amount=d["amount"], weight=d["weight"],
                   curve=HelmCurve.from_dict(d["curve"]) if d["curve"] is not None else None,
                   stage=d["stage"], split=d["split"], info=d["info"])


@dataclass(frozen=True, eq=False)
class Helm:
    """The `HELM` type: an ordered, immutable list of axes plus any warnings gathered while building it."""
    axes: tuple
    warnings: tuple = ()

    def __post_init__(self):
        axes = tuple(self.axes)
        if not 1 <= len(axes) <= MAX_AXES:
            raise HelmValueError(f"a HELM holds 1 to {MAX_AXES} axes (got {len(axes)})")
        for a in axes:
            if not isinstance(a, HelmAxis):
                raise HelmValueError("a HELM holds only HelmAxis entries")
        warns = tuple(_text(w, "warning", 500) for w in self.warnings)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "warnings", warns[:32])

    @classmethod
    def of(cls, axis, parent=None):
        """A new HELM: the parent's axes (if any) followed by `axis`."""
        base = tuple(parent.axes) if parent is not None else ()
        warns = tuple(parent.warnings) if parent is not None else ()
        return cls(base + (axis,), warns)

    def map_axes(self, fn):
        return Helm(tuple(fn(a) for a in self.axes), self.warnings)

    def with_warnings(self, *more):
        return Helm(self.axes, self.warnings + tuple(more))

    def to_dict(self):
        return {"helm_version": HELM_VERSION, "axes": [a.to_dict() for a in self.axes],
                "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, d):
        _exact_keys(d, {"helm_version", "axes", "warnings"}, "helm")
        if d["helm_version"] != HELM_VERSION:
            raise HelmValueError(f"unsupported helm_version {d['helm_version']!r} (this build reads {HELM_VERSION})")
        if not isinstance(d["axes"], list) or not isinstance(d["warnings"], list):
            raise HelmValueError("axes and warnings must be lists")
        return cls(tuple(HelmAxis.from_dict(a) for a in d["axes"]), tuple(d["warnings"]))


def validate_helm(value, what="helm"):
    """Return `value` if it is a HELM, else raise a plain-English error naming the input."""
    if not isinstance(value, Helm):
        raise HelmValueError(f"'{what}' must be a HELM from a Helm · Axis, Shape or Mix node "
                             f"(got {type(value).__name__})")
    return value


# --- mixing -----------------------------------------------------------------------------------
def mix_axes(axes, orthogonalize=True):
    """Pure. Orthogonalize the ACTIVE axes that share a model, block and width, in order (earlier wins).

    Returns (new_axes, warnings). Axes that are inactive (weight 0 or amount 0) are left alone and never
    influence the others. Axes for a different model or block than the first are not comparable and are
    reported, not mixed. A dependent axis (nothing left after projection) is kept but reported."""
    axes = list(axes)
    warns = []
    keys = {}
    for i, a in enumerate(axes):
        keys.setdefault((a.model, a.block, a.dim), []).append(i)
    if len({k[0] for k in keys}) > 1:
        warns.append("These axes were made for different models (" +
                     ", ".join(sorted({str(k[0]) for k in keys})) + "). Only axes for the connected model can be applied.")
    if len({k[1] for k in keys}) > 1:
        warns.append("These axes live at different blocks (" +
                     ", ".join(sorted({str(k[1]) for k in keys})) +
                     "). They are applied separately and cannot be made independent of each other.")
    if not orthogonalize:
        return axes, warns
    out = list(axes)
    for (model, block, dim), idx in keys.items():
        act = [i for i in idx if axes[i].is_active]
        if len(act) < 2:
            continue
        new, dropped = gram_schmidt([axes[i].vec for i in act])
        for pos, i in enumerate(act):
            if new[pos] is None:
                warns.append(f"'{axes[i].label}' is almost entirely contained in the axes before it, so there is "
                             f"nothing independent left of it. It is applied unchanged.")
                continue
            if pos > 0:
                info = dict(axes[i].info)
                info["orthogonalized"] = True
                out[i] = axes[i].with_changes(vec=new[pos], info=info)
    return out, warns
