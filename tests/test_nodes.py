import types

import torch
import pytest

import conftest


def _fake_model():
    return conftest.fake_model(conftest.make_dit(dim=8, n=6))


def _h3_model():
    class MiniMaxH3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    return conftest.FakePatcher(types.SimpleNamespace(diffusion_model=MiniMaxH3()))


def test_apply_warns_when_axis_has_no_safe_max(helm, capsys):
    helm.nodes._save_axis("loose", torch.ones(8), {"kind": "axis", "dim": 8, "block": 0})
    helm.nodes.HelmApply().run(_fake_model(), "loose", 0, 0.1)
    out = capsys.readouterr().out
    assert "no safe_max" in out
    assert "unclamped" in out


def test_dial_without_safe_max_is_reported(helm, capsys):
    helm.nodes._save_axis("loose", torch.ones(8), {"kind": "axis", "dim": 8, "block": 0})
    helm.nodes.HelmApply().run(_fake_model(), "loose", 0, 0.1, dial=0.5)
    out = capsys.readouterr().out
    assert "dial=0.5 ignored" in out


def test_apply_refuses_axis_dim_mismatch_before_sampling(helm):
    helm.nodes._save_axis("wrong", torch.ones(8), {"kind": "axis", "dim": 8, "block": 0})
    with pytest.raises(ValueError, match="calibrated model dim 5376"):
        helm.nodes.HelmApply().run(_h3_model(), "wrong", 0, 0.1)


def test_helm_vector_refuses_mismatched_blocks(helm):
    helm.nodes._save_axis("p", torch.ones(6), {"kind": "axis", "dim": 6, "block": 20, "model": "ZImage"})
    helm.nodes._save_axis("q", torch.ones(6), {"kind": "axis", "dim": 6, "block": 12, "model": "ZImage"})
    with pytest.raises(ValueError):
        helm.nodes.HelmVector().run("p", "q", "axis")
