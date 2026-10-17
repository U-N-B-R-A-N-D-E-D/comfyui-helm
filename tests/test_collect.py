"""Collect write policy: one write per run, no cross-run accumulation (spec 1.6 / NOISE#4)."""
import types

import torch

import conftest
from conftest import FakePatcher, make_c_in, make_dit


def test_one_write_per_run_and_acc_resets(helm, monkeypatch):
    nodes = helm.nodes
    dit = make_dit(dim=8, n=6)
    holder = {"x": torch.full((1, 8), 1.0)}

    def apply_model(x, t, **kwargs):
        return dit(holder["x"])

    saves = []

    def fake_save(name, tensor, meta):
        saves.append((tensor.detach().clone(), str(name)))

    monkeypatch.setattr(nodes, "_save_axis", fake_save)

    mp = nodes.HelmCollect().run(FakePatcher(types.SimpleNamespace(diffusion_model=dit)), "pos", 2)[0]
    wrapper = mp.model_options["model_function_wrapper"]
    sig = [1.0, 0.5, 0.0]  # total = 2
    c_in = make_c_in(torch.zeros(1, 8), sig=sig)

    wrapper(apply_model, c_in)  # step 0
    wrapper(apply_model, c_in)  # step 1 (last) -> one write for run 1
    first_run = [t for t, p in saves if p == "pos"]
    assert len(first_run) == 1, "expected exactly one axis write per run"

    saves.clear()
    holder["x"] = torch.full((1, 8), 9.0)
    wrapper(apply_model, c_in)  # new run: step must reset to 0
    wrapper(apply_model, c_in)
    second_run = [t for t, p in saves if p == "pos"]
    assert len(second_run) == 1, "expected exactly one axis write on the second run"
    assert torch.allclose(second_run[0], torch.full((8,), 9.0)), "acc leaked across runs"
