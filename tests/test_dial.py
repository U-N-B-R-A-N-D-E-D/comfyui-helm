"""HelmDial: human menu, amount -> strength mapping, plain-English model mismatch errors."""
import json
import os
import types

import pytest
import torch

import conftest
from conftest import FakePatcher, make_c_in, make_dit

NODE = "FakeDiT"  # class name of conftest.FakeDiT


def _model(dit=None):
    return conftest.fake_model(dit or make_dit(dim=8, n=6))


def _axis(helm, name, vec=None, **meta):
    base = {"kind": "axis", "dim": 8, "block": 2, "model": NODE, "safe_max": 0.4}
    base.update(meta)
    base = {k: v for k, v in base.items() if v is not None}
    helm.nodes._save_axis(name, torch.arange(8, dtype=torch.float32) if vec is None else vec, base)


def _run_wrapper(mp, x):
    dit = mp.model.diffusion_model
    wrapper = mp.model_options["model_function_wrapper"]
    return wrapper(conftest.make_apply_model(dit), make_c_in(x))


# --- amount -> strength -------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected", [(0.0, 0.0), (0.25, 0.1), (0.5, 0.2), (1.0, 0.4), (2.0, 0.4), (-1.0, 0.0)])
def test_amount_maps_to_zero_through_safe_max(helm, amount, expected):
    assert helm.nodes._dial_strength(amount, 0.4) == pytest.approx(expected)


def test_dial_refuses_an_axis_without_a_safe_range(helm):
    with pytest.raises(ValueError, match="no safe range"):
        helm.nodes._dial_strength(0.5, None)


def test_dial_injects_amount_times_safe_max(helm):
    _axis(helm, "ax", safe_max=0.4)
    mp = helm.nodes.HelmDial().run(_model(), "Custom · ax", 0.5)[0]
    x = torch.zeros(1, 8)
    out = _run_wrapper(mp, x)
    assert torch.allclose(out, 0.2 * torch.arange(8, dtype=torch.float32).reshape(1, 8))


def test_dial_amount_zero_is_a_passthrough(helm):
    _axis(helm, "ax")
    mp = helm.nodes.HelmDial().run(_model(), "Custom · ax", 0.0)[0]
    x = torch.randn(1, 8)
    assert torch.equal(_run_wrapper(mp, x), x)


def test_dial_uses_the_block_recorded_in_the_axis(helm):
    _axis(helm, "ax", block=4)
    mp = helm.nodes.HelmDial().run(_model(), "Custom · ax", 1.0)[0]
    assert [h["block"] for h in mp.model_options["helm_hooks"]] == [4]


def test_dial_reuses_helmapply_hook(helm):
    """One hook implementation: Dial at amount a == HelmApply at strength a*safe_max."""
    _axis(helm, "ax", safe_max=0.4)
    x = torch.randn(1, 8)
    a = _run_wrapper(helm.nodes.HelmDial().run(_model(), "Custom · ax", 0.75)[0], x)
    b = _run_wrapper(helm.nodes.HelmApply().run(_model(), "ax", 2, 0.3)[0], x)
    assert torch.allclose(a, b)


def test_axis_from_overrides_the_menu(helm):
    _axis(helm, "mine", safe_max=0.8)
    mp = helm.nodes.HelmDial().run(_model(), "(no axes yet - build one with Helm · Make Axis)", 1.0, axis_from="mine")[0]
    out = _run_wrapper(mp, torch.zeros(1, 8))
    assert torch.allclose(out, 0.8 * torch.arange(8, dtype=torch.float32).reshape(1, 8))


# --- model mismatch -----------------------------------------------------------------------

def test_mismatched_model_gives_a_plain_message(helm):
    _axis(helm, "zimg", model="NextDiT")
    with pytest.raises(ValueError) as e:
        helm.nodes.HelmDial().run(_model(), "Custom · zimg", 0.5)
    msg = str(e.value)
    assert "made for Z-Image" in msg and "Make Axis" in msg
    assert "strength" not in msg.lower() and "norm" not in msg.lower()


def test_dimension_mismatch_on_a_calibrated_model_is_plain(helm):
    class MiniMaxH3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    _axis(helm, "small", model=None)  # dim 8, no model recorded
    h3 = FakePatcher(types.SimpleNamespace(diffusion_model=MiniMaxH3Model()))
    with pytest.raises(ValueError, match="does not fit this model"):
        helm.nodes.HelmDial().run(h3, "Custom · small", 0.5)


def test_missing_axis_file_says_how_to_get_it(helm, monkeypatch):
    monkeypatch.setattr(helm.nodes, "_READ_DIRS", [helm.steer, helm.models_steer])  # no shipped axes
    with pytest.raises(FileNotFoundError, match="Essentials"):
        helm.nodes.HelmDial().run(_model(), "Essentials · Light angle (raking) (Z-Image)", 0.5)


def test_no_axes_at_all_points_to_make_axis(helm):
    with pytest.raises(ValueError, match="Make Axis"):
        helm.nodes._dial_resolve(helm.nodes._NO_AXES)


def test_validate_inputs_accepts_an_axis_that_does_not_exist_yet(helm):
    # Make Axis writes the file during the run, after the prompt was validated.
    assert helm.nodes.HelmDial.VALIDATE_INPUTS("Custom · not_yet_made") is True
    assert helm.nodes.HelmDial.VALIDATE_INPUTS("") != True  # noqa: E712


# --- the menu -----------------------------------------------------------------------------

def test_menu_lists_essentials_then_subtle_then_custom(helm):
    _axis(helm, "my_fog")
    choices = helm.nodes._dial_choices()
    assert "Essentials · Less clutter (Z-Image)" in choices
    assert "Essentials · Less clutter (MiniMax H3)" in choices
    assert "Subtle · Motion blur (Z-Image)" in choices
    assert "Custom · my_fog" in choices
    assert "Custom · mood" in choices  # shipped but not benchmarked -> Custom, never Essentials
    sections = [c.split(" · ")[0] for c in choices]
    assert sections == sorted(sections, key=["Essentials", "Subtle", "Custom"].index), "section order"


def test_menu_hides_recordings_and_absent_catalog_axes(helm, monkeypatch):
    monkeypatch.setattr(helm.nodes, "_READ_DIRS", [helm.steer, helm.models_steer])  # no shipped axes
    helm.nodes._save_axis("pos", torch.ones(8), {"kind": "activation", "dim": 8, "block": 2})
    helm.nodes._save_axis("probe_pos", torch.ones(3, 8), {"kind": "probe", "dim": 8, "n_blocks": 3})
    choices = " | ".join(helm.nodes._dial_choices())
    assert not any(c.endswith(("pos", "probe_pos")) for c in helm.nodes._dial_choices())
    assert "Light angle" not in choices  # catalog axis whose file is not installed is not offered


def test_catalog_axis_appears_once_its_file_exists(helm):
    helm.nodes._save_axis("z_raking_light_angle", torch.ones(3840), {
        "kind": "axis", "dim": 3840, "block": 15, "model": "NextDiT"})
    assert "Essentials · Light angle (raking) (Z-Image)" in helm.nodes._dial_choices()


def test_catalog_fallback_safe_max_is_used_when_the_header_has_none(helm):
    helm.nodes._save_axis("z_raking_light_angle", torch.ones(8), {
        "kind": "axis", "dim": 8, "block": 2, "model": NODE})  # no safe_max
    mp = helm.nodes.HelmDial().run(_model(), "Essentials · Light angle (raking) (Z-Image)", 1.0)[0]
    out = _run_wrapper(mp, torch.zeros(1, 8))
    assert torch.allclose(out, torch.full((1, 8), 2.8064))


def test_dial_is_the_first_node_in_the_helm_category(helm):
    n = helm.nodes
    assert n.HelmDial.CATEGORY == "Helm" and n.HelmMakeAxis.CATEGORY == "Helm"
    for cls in ("HelmCollect", "HelmVector", "HelmApply", "HelmProbe", "HelmProbeReport"):
        assert n.NODE_CLASS_MAPPINGS[cls].CATEGORY == "Helm/Lab"
        assert n.NODE_DISPLAY_NAME_MAPPINGS[cls].startswith("Helm Lab")


def test_tooltips_are_present_and_plain(helm):
    spec = helm.nodes.HelmDial.INPUT_TYPES()
    for section in spec.values():
        for name, cfg in section.items():
            tip = cfg[1]["tooltip"]
            assert len(tip) > 20, name
            for jargon in ("lambda", "diff-in-means", "safe_max", "block "):
                assert jargon not in tip.lower(), (name, jargon)


# --- the catalog is tied to the repo's own benchmark ----------------------------------------

def test_catalog_matches_the_study_data(helm):
    import csv
    rows = {r["axis"]: r for r in csv.DictReader(open(os.path.join(helm.root, "study", "data", "axes.csv")))}
    catalog = json.load(open(os.path.join(helm.root, "axes_catalog.json")))["axes"]
    z = [a for a in catalog if a["file"] not in ("z_room_density", "density")]
    assert len(z) == 7
    for a in z:
        r = rows[a["file"]]
        assert r["strict_push"] == "holds" and r["monotone"] == "yes", a["file"]
        assert a["fallback_safe_max"] == pytest.approx(float(r["top_strength"])), a["file"]
    for a in catalog:
        assert a["shipped"]
        assert a["evidence"] and a["label"] and a["family"]
        if a["shipped"]:
            assert os.path.exists(os.path.join(helm.root, "example_axes", a["file"] + ".safetensors"))
            meta = helm.nodes._load_axis_meta(os.path.join(helm.root, "example_axes", a["file"] + ".safetensors"))
            assert meta["model"] == a["model_class"]
            assert meta["safe_max"] == pytest.approx(a["fallback_safe_max"])


# --- Essentials: every entry ships, is the file we recorded, and resolves a safe range ---------

def _provenance():
    rows = {}
    text = open(os.path.join(conftest.ROOT, "docs", "AXES.md"), encoding="utf-8").read()
    for line in text.splitlines():
        parts = [p.strip().strip("`") for p in line.strip().strip("|").split("|")]
        if len(parts) == 4 and parts[0].startswith("z_"):
            rows[parts[0]] = (float(parts[1]), parts[2], parts[3])
    return rows


SUBTLE_FILES = {"z_volumetric_beams", "z_surface_reflection_intensity", "z_motion_blur"}


def _catalog(helm):
    return json.load(open(os.path.join(helm.root, "axes_catalog.json")))["axes"]


def test_every_menu_entry_has_its_file_and_a_resolvable_safe_max(helm):
    entries = [e for e in helm.nodes._dial_entries() if e["section"] in ("Essentials", "Subtle")]
    catalog = _catalog(helm)
    assert sorted(e["file"] for e in entries) == sorted(a["file"] for a in catalog)
    assert len(entries) == 9
    for e in entries:
        path = helm.nodes._resolve_axis(e["file"])
        assert path.startswith(os.path.join(helm.pack, "example_axes"))
        assert e["safe_max"] and e["safe_max"] > 0
        assert helm.nodes._dial_strength(1.0, e["safe_max"]) == pytest.approx(e["safe_max"])


def test_subtle_section_holds_exactly_the_three_axes_with_little_visible_effect(helm):
    by = {e["file"]: e["section"] for e in helm.nodes._dial_entries()}
    assert {f for f, sec in by.items() if sec == "Subtle"} == SUBTLE_FILES
    assert {f for f, sec in by.items() if sec == "Essentials"} == {
        "z_corridor_orientation", "z_aisle_width", "z_overcast_vs_direct", "z_raking_light_angle",
        "z_room_density", "density"}


def test_essentials_are_ordered_by_visible_impact(helm):
    """visible_impact = top-dose saturation + peak luminance shift + |edge ratio - 1|, all from axes.csv."""
    import csv
    rows = {r["axis"]: r for r in csv.DictReader(open(os.path.join(helm.root, "study", "data", "axes.csv")))}

    def score(f):
        r = rows[f]
        return round(float(r["sat_top"]) + float(r["lum_delta_max"]) + abs(float(r["edge_ratio_top"]) - 1), 3)

    catalog = _catalog(helm)
    for a in catalog:
        if a["file"] in rows:
            assert a["visible_impact"] == pytest.approx(score(a["file"])), a["file"]
    ess = [a for a in catalog if a["tier"] == "essential" and "visible_impact" in a]
    assert [a["visible_impact"] for a in ess] == sorted((a["visible_impact"] for a in ess), reverse=True)
    sub = [a for a in catalog if a["tier"] == "subtle"]
    assert [a["visible_impact"] for a in sub] == sorted((a["visible_impact"] for a in sub), reverse=True)
    assert max(a["visible_impact"] for a in sub) < min(a["visible_impact"] for a in ess)
    # the menu follows that order, Essentials first, then Subtle
    menu = [e["file"] for e in helm.nodes._dial_entries() if e["section"] != "Custom"]
    assert menu == [a["file"] for a in sorted(catalog, key=lambda a: a["tier"] == "subtle")]


def test_new_dial_starts_on_a_plain_readable_idea(helm):
    default = helm.nodes.HelmDial.INPUT_TYPES()["required"]["axis"][1]["default"]
    assert default == "Essentials · Light angle (raking) (Z-Image)"
    assert helm.nodes._dial_default(["Custom · x"]) == "Custom · x"


@pytest.mark.parametrize("amount,words", [(0.6, "Light angle (raking) at 60%"), (0.0, "Light angle (raking) at 0% (Dial off)"),
                                           (1.0, "Light angle (raking) at 100%"), (7, "Light angle (raking) at 100%")])
def test_dial_says_what_it_does_in_plain_words(helm, amount, words):
    assert helm.nodes._dial_words("Light angle (raking)", amount) == words


def test_dial_logs_the_plain_words(helm, capsys):
    _axis(helm, "ax", safe_max=0.4)
    helm.nodes.HelmDial().run(_model(), "Custom · ax", 0.6)
    assert "Dial: ax at 60%" in capsys.readouterr().out


def test_every_input_of_every_node_has_a_tooltip_and_every_node_a_description(helm):
    for name, cls in helm.nodes.NODE_CLASS_MAPPINGS.items():
        assert len(cls.DESCRIPTION) > 30, name
        assert cls.SEARCH_ALIASES, name
        # Lab nodes are found by what they do; only the two main nodes carry the pack name, so they rank first
        assert ("helm" in " ".join(cls.SEARCH_ALIASES).lower()) == (cls.CATEGORY == "Helm"), name
        for section in cls.INPUT_TYPES().values():
            for inp, cfg in section.items():
                assert len(cfg) > 1 and len(cfg[1].get("tooltip", "")) > 8, (name, inp)


def test_shipped_z_image_axes_match_the_provenance_table(helm):
    import hashlib
    import struct
    prov = _provenance()
    assert len(prov) == 7
    for name, (safe, file_sha, tensor_sha) in prov.items():
        raw = open(os.path.join(helm.pack, "example_axes", name + ".safetensors"), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == file_sha, name
        n = struct.unpack("<Q", raw[:8])[0]
        assert hashlib.sha256(raw[8 + n:]).hexdigest() == tensor_sha, name  # tensor bytes as captured
        meta = helm.nodes._load_axis_meta(os.path.join(helm.pack, "example_axes", name + ".safetensors"))
        assert meta["safe_max"] == pytest.approx(safe)
        assert meta["safe_max_source"] == "benchmark (axes.csv)"
        assert meta["model"] == "NextDiT" and meta["block"] == 15 and meta["dim"] == 3840
        assert meta["from"] == {"pos": name + "_pos", "neg": name + "_neg"}  # tags only, no paths


def test_header_safe_max_equals_the_benchmark_top_strength(helm):
    import csv
    rows = {r["axis"]: r for r in csv.DictReader(open(os.path.join(helm.root, "study", "data", "axes.csv")))}
    for name, (safe, _f, _t) in _provenance().items():
        assert safe == pytest.approx(float(rows[name]["top_strength"]))
