"""Windowing: hooks outside [start_percent, end_percent] are skipped (spec 1.4 / 1.7)."""
import types

import torch

import conftest
from conftest import FakePatcher, make_apply_model, make_c_in, make_dit


def test_hook_skipped_before_start_and_runs_inside(helm):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    mp = FakePatcher(types.SimpleNamespace(diffusion_model=dit))
    calls = []

    def fn(m, args, out, flags=None):
        calls.append(1)
        x = out[0] if isinstance(out, tuple) else out
        return x + 1.0

    nodes._add_hook(mp, {"block": 2, "fn": fn, "cond_only": False, "start": 0.5, "end": 1.0})
    wrapper = nodes._wrapper(mp)
    x = torch.zeros(1, 8, dtype=torch.float32)
    sig = [1.0, 0.5, 0.0]  # total = 2 -> frac at step0=0.0, step1=0.5

    out0 = wrapper(make_apply_model(dit), make_c_in(x, sig=sig))
    assert calls == [], "hook ran before start_percent"
    assert torch.allclose(out0, x)

    out1 = wrapper(make_apply_model(dit), make_c_in(x, sig=sig))
    assert len(calls) == 1, "hook did not run inside the window"
    assert torch.allclose(out1, x + 1.0)


def test_hook_skipped_after_end(helm):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    mp = FakePatcher(types.SimpleNamespace(diffusion_model=dit))
    calls = []

    def fn(m, args, out, flags=None):
        calls.append(1)
        x = out[0] if isinstance(out, tuple) else out
        return x + 1.0

    nodes._add_hook(mp, {"block": 2, "fn": fn, "cond_only": False, "start": 0.0, "end": 0.4})
    wrapper = nodes._wrapper(mp)
    x = torch.zeros(1, 8, dtype=torch.float32)
    sig = [1.0, 0.5, 0.0]

    wrapper(make_apply_model(dit), make_c_in(x, sig=sig))  # step0 frac 0 -> runs
    assert len(calls) == 1
    out1 = wrapper(make_apply_model(dit), make_c_in(x, sig=sig))  # step1 frac .5 > .4 -> skipped
    assert len(calls) == 1, "hook ran after end_percent"
    assert torch.allclose(out1, x)


def test_no_sigmas_total_unknown_runs_every_step(helm):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    mp = FakePatcher(types.SimpleNamespace(diffusion_model=dit))
    calls = []

    def fn(m, args, out, flags=None):
        calls.append(1)
        x = out[0] if isinstance(out, tuple) else out
        return x + 1.0

    nodes._add_hook(mp, {"block": 2, "fn": fn, "cond_only": False, "start": 0.0, "end": 1.0})
    wrapper = nodes._wrapper(mp)
    x = torch.zeros(1, 8, dtype=torch.float32)
    wrapper(make_apply_model(dit), make_c_in(x, sig=None))
    wrapper(make_apply_model(dit), make_c_in(x, sig=None))
    assert len(calls) == 2
