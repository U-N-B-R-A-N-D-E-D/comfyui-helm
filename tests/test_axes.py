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
    assert resolved == os.path.join(helm.pack, "example_axes", "mood.safetensors")


def test_resolution_prefers_safetensors_over_pt_in_the_same_dir(helm):
    _save(os.path.join(helm.steer, "both.pt"), torch.ones(3))
    helm.nodes._save_axis("both", torch.zeros(3) + 5, {"kind": "axis", "dim": 3})
    assert helm.nodes._resolve_axis("both") == os.path.join(helm.steer, "both.safetensors")


def test_legacy_pt_is_still_readable(helm):
    _save(os.path.join(helm.steer, "old.pt"), torch.arange(4, dtype=torch.float32))
    vec = helm.nodes._load_axis(helm.nodes._resolve_axis("old"))
    assert torch.equal(vec, torch.arange(4, dtype=torch.float32))
    assert vec.dtype == torch.float32


def test_meta_from_header_casts_types(helm):
    header = {"kind": "axis", "dim": "5376", "block": "20", "norm": "3170.8",
              "safe_max": "0.3", "model": "", "from": '{"pos": "a", "neg": "b"}'}
    meta = helm.nodes._meta_from_header(header)
    assert meta["dim"] == 5376 and meta["block"] == 20
    assert meta["norm"] == 3170.8 and meta["safe_max"] == 0.3
    assert meta["model"] is None
    assert meta["from"] == {"pos": "a", "neg": "b"}


def test_axis_file_is_self_describing_without_the_sidecar(helm):
    helm.nodes._save_axis("lone", torch.ones(6), {"kind": "axis", "dim": 6, "block": 12,
                                                  "model": "MiniMaxH3Model", "module": "blocks"})
    os.remove(os.path.join(helm.steer, "lone.json"))
    meta = helm.nodes._read_meta("lone")
    assert meta["block"] == 12 and meta["dim"] == 6 and meta["module"] == "blocks"


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


def test_a_users_axis_is_not_described_by_the_packs_sidecar(helm):
    """D2: a `.safetensors` dropped in user/steer/ must be described by its OWN header, never by a
    same-named sidecar that happens to live in another read directory."""
    import os
    helm.nodes._save_axis("mood", torch.ones(6), {"kind": "axis", "dim": 6, "block": 15,
                                                  "model": "ZImage", "module": "layers"})
    assert os.path.exists(os.path.join(helm.pack, "example_axes", "mood.json"))
    meta = helm.nodes._read_meta("mood")
    assert meta["block"] == 15, f"took metadata from another directory: {meta}"
    assert meta["module"] == "layers"


def test_an_axis_with_no_metadata_never_inherits_another_directory(helm):
    """Audit 4.1.1: the old name-match fallback is gone. A file with no sidecar beside it and no
    header must have NO metadata -- never example_axes/mood.json's block."""
    import os
    from safetensors.torch import save_file
    os.makedirs(helm.user, exist_ok=True)
    save_file({"axis": torch.ones(6)}, os.path.join(helm.user, "mood.safetensors"))
    assert os.path.exists(os.path.join(helm.pack, "example_axes", "mood.json"))
    assert helm.nodes._read_meta("mood") is None
