"""Axis resolution order and sidecar meta (spec 1.1 / 1.2)."""
import os

import pytest
import torch


def _save(path, vec):
    torch.save(vec, path)


def test_resolution_prefers_write_dir(helm):
    p = os.path.join(helm.steer, "ax_user.pt")
    _save(p, torch.zeros(4))
    assert helm.nodes._resolve_axis("ax_user") == p


def test_resolution_falls_back_to_models_dir(helm):
    p = os.path.join(helm.models_steer, "ax_models.pt")
    _save(p, torch.zeros(4))
    assert helm.nodes._resolve_axis("ax_models") == p


def test_resolution_falls_back_to_example_axes(helm):
    resolved = helm.nodes._resolve_axis("mood")
    assert resolved == os.path.join(helm.pack, "example_axes", "mood.pt")


def test_resolution_order_user_beats_models_and_pack(helm):
    _save(os.path.join(helm.models_steer, "mood.pt"), torch.ones(3))
    _save(os.path.join(helm.steer, "mood.pt"), torch.ones(3) * 2)
    assert helm.nodes._resolve_axis("mood") == os.path.join(helm.steer, "mood.pt")


def test_missing_axis_error_lists_every_searched_dir(helm):
    with pytest.raises(FileNotFoundError) as ei:
        helm.nodes._resolve_axis("does_not_exist_anywhere")
    msg = str(ei.value)
    for d in (helm.steer, helm.models_steer, os.path.join(helm.pack, "example_axes")):
        assert d in msg
    assert "does_not_exist_anywhere" in msg


def test_resolve_meta_follows_axis_dirs(helm):
    p = os.path.join(helm.models_steer, "ax_meta.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"kind": "axis", "dim": 4}')
    meta = helm.nodes._resolve_meta("ax_meta")
    assert meta == p or str(meta) == p


def test_missing_meta_lists_every_searched_dir(helm):
    with pytest.raises(FileNotFoundError) as ei:
        helm.nodes._resolve_meta("no_such_meta")
    msg = str(ei.value)
    for d in (helm.steer, helm.models_steer, os.path.join(helm.pack, "example_axes")):
        assert d in msg


def test_write_meta_preserves_created_and_sets_version(helm):
    helm.nodes._write_meta("mk_keep", {"kind": "axis", "created": 123})
    meta = helm.nodes._read_meta("mk_keep")
    assert meta["created"] == 123
    assert meta["helm_version"] == 2
    assert meta["kind"] == "axis"


def test_write_meta_defaults_created_and_version(helm):
    helm.nodes._write_meta("mk_fresh", {"kind": "axis"})
    meta = helm.nodes._read_meta("mk_fresh")
    assert isinstance(meta["created"], (int, float))
    assert meta["helm_version"] == 2


def test_write_meta_does_not_assign_none_over_value(helm):
    helm.nodes._write_meta("mk_dev", {"created": 7.5, "block": 20})
    meta = helm.nodes._read_meta("mk_dev")
    assert meta["created"] == 7.5
