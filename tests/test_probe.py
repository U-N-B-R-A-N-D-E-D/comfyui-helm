"""HelmProbe / HelmProbeReport: per-block sensitivity profile for onboarding a new model."""
import types

import pytest
import torch

from conftest import FakePatcher, make_c_in

BAND = 12  # a block inside the middle 50% of 20 (where the suggestion is allowed to land)


def _planted(n_blocks=20, dim=8, seed=0):
    """pos/neg means that differ only at one block, so the 'right' answer is known."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_blocks, dim, generator=g)
    pos = base.clone()
    neg = base.clone()
    pos[BAND] += torch.randn(dim, generator=g) * 5.0
    return pos, neg


def test_probe_profile_finds_the_planted_block(helm):
    prof = helm.nodes._probe_profile(*_planted())
    assert prof["suggested"]["block"] == BAND
    assert prof["ranking"][0]["block"] == BAND
    assert prof["dim"] == 8 and prof["n_blocks"] == 20
    assert prof["suggested"]["strength"] > 0


def test_probe_profile_separation_is_scale_free(helm):
    pos, neg = _planted()
    a = helm.nodes._probe_profile(pos, neg)
    b = helm.nodes._probe_profile(pos * 100.0, neg * 100.0)
    assert a["suggested"]["separation"] == b["suggested"]["separation"]
    assert a["suggested"]["block"] == b["suggested"]["block"]


def test_probe_report_round_trip(helm):
    pos, neg = _planted()
    helm.nodes._save_tensor("probe_pos", pos, {"kind": "probe", "n_blocks": 20, "dim": 8,
                                               "model": "FakeDiT", "module": "blocks"})
    helm.nodes._save_tensor("probe_neg", neg, {"kind": "probe", "n_blocks": 20, "dim": 8,
                                               "model": "FakeDiT", "module": "blocks"})
    block, strength = helm.nodes.HelmProbeReport().run("probe_pos", "probe_neg", 3)
    assert block == BAND and strength > 0
    with open(f"{helm.steer}/probe_report.json", encoding="utf-8") as f:
        import json
        report = json.load(f)
    assert report["suggested"]["block"] == BAND and report["model"] == "FakeDiT"
    assert report["heuristic"]["safe_fraction"] == helm.nodes.SAFE_FRACTION


def test_probe_report_rejects_mismatched_shapes(helm):
    helm.nodes._save_tensor("probe_pos", torch.ones(20, 8), {"kind": "probe"})
    helm.nodes._save_tensor("probe_neg", torch.ones(20, 4), {"kind": "probe"})
    with pytest.raises(ValueError):
        helm.nodes.HelmProbeReport().run("probe_pos", "probe_neg", 3)


class _Const(torch.nn.Module):
    """Each block emits its own constant, so the expected per-block mean is exactly `n`."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.zeros_like(x) + float(self.n)


def _apply_model(dit):
    def fn(x, t, **kwargs):
        for b in dit.blocks:
            x = b(x)
        return x

    return fn


def test_probe_hook_writes_once_per_run_and_averages_over_steps(helm, monkeypatch):
    nodes = helm.nodes
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Const(i + 1) for i in range(6)])
    mp = FakePatcher(types.SimpleNamespace(diffusion_model=dit))

    acc = {"rows": None, "steps": 0, "n_blocks": 6, "last": False}
    acc["reset"] = lambda: acc.update(rows=None, steps=0)

    saved = []

    def fake_save_tensor(name, tensor, meta, key="axis"):
        saved.append(tensor.detach().clone())

    monkeypatch.setattr(nodes, "_save_tensor", fake_save_tensor)

    fn = nodes._make_probe_hook(acc, "probe_pos", "blocks", "Fake", False, "last")
    nodes._add_hook(mp, {"fn": fn, "all_blocks": True, "start": 0.0, "end": 1.0})
    wrapper = nodes._wrapper(mp)

    c_in = make_c_in(torch.zeros(1, 8), sig=[1.0, 0.5, 0.0])
    wrapper(_apply_model(dit), c_in)  # step 0
    wrapper(_apply_model(dit), c_in)  # step 1 == last -> exactly one write

    assert len(saved) == 1, "expected exactly one probe write per run"
    rows = saved[0]
    assert tuple(rows.shape) == (6, 8)
    for i in range(6):
        assert torch.allclose(rows[i], torch.full((8,), float(i + 1))), f"block {i} mean is wrong"


def test_probe_rows_buffer_follows_the_activation_device(helm, monkeypatch):
    """Regression: the rows buffer was allocated on the *default* device, so a CUDA activation hit
    "Expected all tensors to be on the same device" on the pod on the very first block. CPU-only
    tests cannot reproduce a device mismatch, so pin the root cause instead: the buffer must be
    allocated with the activation's own device."""
    nodes = helm.nodes
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Const(i + 1) for i in range(6)])
    mp = FakePatcher(types.SimpleNamespace(diffusion_model=dit))

    seen = {}
    real_zeros = torch.zeros

    def spy_zeros(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(nodes.torch, "zeros", spy_zeros)

    acc = {"rows": None, "steps": 0, "n_blocks": 6, "last": False}
    acc["reset"] = lambda: acc.update(rows=None, steps=0)
    fn = nodes._make_probe_hook(acc, "probe_pos", "blocks", "Fake", False, "last")
    nodes._add_hook(mp, {"fn": fn, "all_blocks": True, "start": 0.0, "end": 1.0})

    x = torch.zeros(1, 8)
    nodes._wrapper(mp)(_apply_model(dit), make_c_in(x, sig=[1.0, 0.5, 0.0]))

    assert acc["rows"] is not None
    assert seen.get("device") is not None, "probe allocated its rows buffer without a device"
    assert seen["device"] == x.device
