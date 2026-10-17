"""Calibration matching and the LTX module fix (spec 1.8)."""
import json
import os
import types

import torch

import conftest
from conftest import FakePatcher, make_dit


class MiniMaxH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])


class MysteryDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])


def _mp_with(dm):
    return FakePatcher(types.SimpleNamespace(diffusion_model=dm))


def test_calib_for_matches_h3_entry(helm):
    entry = helm.nodes._calib_for(_mp_with(MiniMaxH3()))
    assert entry is not None
    assert entry["module"] == "blocks"
    assert entry["default_block"] == 20


def test_calib_for_unknown_returns_none(helm):
    assert helm.nodes._calib_for(_mp_with(MysteryDiT())) is None


def test_calibration_json_ltx_uses_transformer_blocks(helm):
    with open(os.path.join(helm.pack, "calibration.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    ltx = [m for m in data["models"] if "ltx" in m["match"] or "ltxv" in m["match"]]
    assert ltx, "no LTX entry in calibration.json"
    assert ltx[0]["module"] == "transformer_blocks"
    assert ltx[0]["tested"] is False
