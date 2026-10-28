"""E1 regression: two chained HelmApply must BOTH modify the output, and a foreign wrapper is kept.

Uses a fake ModelPatcher whose `clone()` copies `model_options` exactly like ComfyUI
(`comfy.utils.deepcopy_list_dict`), so a wrapper that is "kept" instead of re-bound to the clone
will silently drop the second hook.
"""
import os
import types

import pytest
import torch

import conftest
from conftest import FakePatcher, make_apply_model, make_c_in, make_dit


def _write_axis(helm, name, vec):
    torch.save(vec, os.path.join(helm.steer, f"{name}.pt"))


def test_block_minus_one_adopts_the_block_recorded_in_the_axis(helm):
    nodes = helm.nodes
    nodes._save_axis("auto", torch.ones(8), {"kind": "axis", "dim": 8, "block": 3})
    dit = make_dit(dim=8, n=6)
    mp = nodes.HelmApply().run(FakePatcher(types.SimpleNamespace(diffusion_model=dit)), "auto", -1, 0.5)[0]
    assert mp.model_options["helm_hooks"][0]["block"] == 3


def test_block_minus_one_without_a_recorded_block_is_an_error(helm):
    nodes = helm.nodes
    nodes._save_axis("nob", torch.ones(8), {"kind": "axis", "dim": 8})
    dit = make_dit(dim=8, n=6)
    with pytest.raises(ValueError):
        nodes.HelmApply().run(FakePatcher(types.SimpleNamespace(diffusion_model=dit)), "nob", -1, 0.5)


def test_two_chained_helmapply_both_change_output(helm):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    base = FakePatcher(types.SimpleNamespace(diffusion_model=dit))

    calls = []

    def foreign(apply_model, c_in):
        calls.append(1)
        return apply_model(c_in["input"], c_in["timestep"], **c_in["c"], cond_or_uncond=c_in["cond_or_uncond"])

    base.model_options["model_function_wrapper"] = foreign

    _write_axis(helm, "axA", torch.eye(8)[0])
    _write_axis(helm, "axB", torch.eye(8)[1])

    m1 = nodes.HelmApply().run(base, "axA", 2, 0.5)[0]
    m2 = nodes.HelmApply().run(m1, "axB", 2, 0.5)[0]
    wrapper = m2.model_options["model_function_wrapper"]

    x = torch.zeros(2, 8, dtype=torch.float32)
    c_in = make_c_in(x, sig=[1.0, 0.5, 0.0], flags=[0, 0])
    out_both = wrapper(make_apply_model(dit), c_in)
    base_out = dit(x)

    dit_a = make_dit(dim=8, n=6)
    mp_a = nodes.HelmApply().run(FakePatcher(types.SimpleNamespace(diffusion_model=dit_a)), "axA", 2, 0.5)[0]
    out_a = mp_a.model_options["model_function_wrapper"](make_apply_model(dit_a), make_c_in(x, sig=[1.0, 0.5, 0.0], flags=[0, 0]))

    dit_b = make_dit(dim=8, n=6)
    mp_b = nodes.HelmApply().run(FakePatcher(types.SimpleNamespace(diffusion_model=dit_b)), "axB", 2, 0.5)[0]
    out_b = mp_b.model_options["model_function_wrapper"](make_apply_model(dit_b), make_c_in(x, sig=[1.0, 0.5, 0.0], flags=[0, 0]))

    assert calls, "pre-existing foreign wrapper was not preserved/called"
    assert not torch.allclose(out_both, base_out), "two chained axes dropped the first one"
    assert not torch.allclose(out_both, out_a), "second HelmApply was a no-op"
    assert not torch.allclose(out_both, out_b), "first HelmApply was a no-op"
    assert torch.allclose(out_both, out_a + out_b - base_out)


def test_add_hook_twice_composes(helm):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    mp = conftest.FakePatcher(types.SimpleNamespace(diffusion_model=dit))

    fn_a = nodes._make_apply_hook(torch.eye(8)[0], 0.5, "axA", False, "metaA")
    fn_b = nodes._make_apply_hook(torch.eye(8)[1], 0.5, "axB", False, "metaB")
    nodes._add_hook(mp, {"block": 2, "fn": fn_a, "cond_only": False, "start": 0.0, "end": 1.0})
    nodes._add_hook(mp, {"block": 2, "fn": fn_b, "cond_only": False, "start": 0.0, "end": 1.0})

    wrapper = nodes._wrapper(mp)
    x = torch.zeros(1, 8, dtype=torch.float32)
    out = wrapper(make_apply_model(dit), make_c_in(x, sig=[1.0, 0.5, 0.0]))
    expected = x + 0.5 * torch.eye(8)[0] + 0.5 * torch.eye(8)[1]
    assert torch.allclose(out, expected)
