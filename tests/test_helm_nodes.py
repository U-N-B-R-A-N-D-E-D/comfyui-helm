"""The Helm 0.5 node family: Axis, Shape, Mix, Apply; chaining, defaults, signed mapping, schedules on the hook
path, back-compat. All on synthetic tensors (no GPU, no ComfyUI)."""
import glob
import json
import os

import pytest
import torch

import conftest
import helm_core as H
from conftest import make_apply_model, make_c_in, make_dit

DIM = 8
SIG = [1.0, 0.5, 0.0]


@pytest.fixture(autouse=True)
def _clear_nuisance_cache(helm):
    helm.nodes._NUISANCE_CACHE.clear()
    yield
    helm.nodes._NUISANCE_CACHE.clear()


def _model(dim=DIM, n=6):
    return conftest.fake_model(make_dit(dim=dim, n=n))


def _vec(k=1):
    v = torch.zeros(DIM)
    v[k - 1] = 1.0
    return v


def _axis(name="a", vec=None, **kw):
    base = dict(name=name, label=name, vec=_vec() if vec is None else vec, model="FakeDiT", block=2, safe_max=0.4)
    base.update(kw)
    return H.HelmAxis(**base)


def _run(mp, x, sig=None, sigma=None, shapes=None):
    wrapper = mp.model_options["model_function_wrapper"]
    c_in = make_c_in(x, sig=sig)
    if sigma is not None:
        c_in["timestep"] = torch.tensor([sigma])
    if shapes is not None:
        c_in["c"]["latent_shapes"] = shapes
    return wrapper(make_apply_model(mp.model.diffusion_model), c_in)


def _save_library_axis(helm, name="ax", **meta):
    base = {"kind": "axis", "dim": DIM, "block": 2, "model": "FakeDiT", "safe_max": 0.4}
    base.update(meta)
    helm.nodes._save_axis(name, torch.arange(1, DIM + 1, dtype=torch.float32), base)


# --- registration and categories ----------------------------------------------------------------
def test_the_four_new_nodes_are_registered_in_the_helm_category(helm):
    n = helm.nodes
    for cls, shown in (("HelmAxis", "Helm · Axis"), ("HelmShape", "Helm · Shape"), ("HelmMix", "Helm · Mix"),
                       ("HelmApplyNode", "Helm · Apply")):
        assert n.NODE_CLASS_MAPPINGS[cls].CATEGORY == "Helm"
        assert n.NODE_DISPLAY_NAME_MAPPINGS[cls] == shown
        assert not getattr(n.NODE_CLASS_MAPPINGS[cls], "DEPRECATED", False)
    assert n.NODE_CLASS_MAPPINGS["HelmDial"].CATEGORY == "Helm"       # the one-node shortcut stays


def test_node_signatures(helm):
    n = helm.nodes
    assert n.HelmAxis.RETURN_TYPES == ("HELM",) and n.HelmShape.RETURN_TYPES == ("HELM",)
    assert n.HelmMix.RETURN_TYPES == ("HELM", "STRING") and n.HelmApplyNode.RETURN_TYPES == ("MODEL",)
    assert n.HelmAxis.INPUT_TYPES()["optional"]["helm"][0] == "HELM"          # inline chaining
    assert n.HelmShape.INPUT_TYPES()["required"]["helm"][0] == "HELM"
    req = n.HelmApplyNode.INPUT_TYPES()["required"]
    assert req["model"][0] == "MODEL" and req["helm"][0] == "HELM"


def test_only_apply_patches_the_model(helm):
    n = helm.nodes
    _save_library_axis(helm)
    mp = _model()
    h = n.HelmAxis().run("library", "Custom · ax", model=mp)[0]
    h = n.HelmShape().run(h, 0.3)[0]
    h, _ = n.HelmMix().run(h)
    assert not mp.model_options                      # nothing so far touched the model
    out = n.HelmApplyNode().run(mp, h)[0]
    assert out.model_options.get("helm_hooks")


# --- Axis: library, defaults, chaining ----------------------------------------------------------
def test_axis_from_library_reads_the_axis_file(helm):
    _save_library_axis(helm, "ax", safe_max=0.4)
    (h,) = helm.nodes.HelmAxis().run("library", "Custom · ax")
    a = h.axes[0]
    assert isinstance(h, H.Helm) and len(h.axes) == 1
    assert (a.label, a.source, a.block, a.safe_max, a.model) == ("ax", "library", 2, 0.4, "FakeDiT")
    assert torch.equal(a.vec, torch.arange(1, DIM + 1, dtype=torch.float32))


def test_axis_refuses_a_library_idea_for_another_model(helm):
    _save_library_axis(helm, "ax", model="SomeOtherModel")
    with pytest.raises(ValueError, match="made for"):
        helm.nodes.HelmAxis().run("library", "Custom · ax", model=_model())


def test_axis_refuses_an_idea_with_no_safe_range(helm):
    helm.nodes._save_axis("nosafe", torch.ones(DIM), {"kind": "axis", "dim": DIM, "block": 2, "model": "FakeDiT"})
    with pytest.raises(ValueError, match="no safe range"):
        helm.nodes.HelmAxis().run("library", "Custom · nosafe")


def test_axis_axis_axis_accumulates_in_order_and_never_mutates_the_input(helm):
    n = helm.nodes
    for name in ("one", "two", "three"):
        _save_library_axis(helm, name)
    (h1,) = n.HelmAxis().run("library", "Custom · one")
    (h2,) = n.HelmAxis().run("library", "Custom · two", helm=h1)
    (h3,) = n.HelmAxis().run("library", "Custom · three", helm=h2)
    assert [a.label for a in h3.axes] == ["one", "two", "three"]
    assert [a.label for a in h1.axes] == ["one"] and [a.label for a in h2.axes] == ["one", "two"]


def test_axis_rejects_a_non_helm_chain_input(helm):
    _save_library_axis(helm)
    with pytest.raises(H.HelmValueError, match="must be a HELM"):
        helm.nodes.HelmAxis().run("library", "Custom · ax", helm="nope")


# --- defaults: Axis -> Apply, no Shape ----------------------------------------------------------
def test_axis_to_apply_with_no_shape_uses_amount_half_whole_clip_all_steps(helm):
    n = helm.nodes
    _save_library_axis(helm, "ax", safe_max=0.4)
    mp = _model()
    h = n.HelmAxis().run("library", "Custom · ax")[0]
    a = h.axes[0]
    assert (a.amount, a.curve, a.stage, a.weight) == (0.5, None, "all", 1.0)
    out = n.HelmApplyNode().run(mp, h)[0]
    base = torch.arange(1, DIM + 1, dtype=torch.float32).reshape(1, DIM)
    for sig, sigma in ((SIG, 1.0), (SIG, 0.5)):                     # every step
        res = _run(out, torch.zeros(1, DIM), sig=sig, sigma=sigma)
        assert torch.allclose(res, 0.5 * 0.4 * base)


def test_the_dial_still_works_as_the_one_node_shortcut(helm):
    n = helm.nodes
    _save_library_axis(helm, "ax", safe_max=0.4)
    dial = n.HelmDial().run(_model(), "Custom · ax", 0.5)[0]
    h = n.HelmAxis().run("library", "Custom · ax")[0]
    new = n.HelmApplyNode().run(_model(), h)[0]
    x = torch.zeros(1, DIM)
    assert torch.allclose(_run(dial, x), _run(new, x))              # same hook, same numbers


# --- signed mapping ------------------------------------------------------------------------------
@pytest.mark.parametrize("amount,factor", [(0.5, 0.5), (1.0, 1.0), (-0.5, -0.5), (-1.0, -1.0), (0.01, 0.01)])
def test_signed_amount_maps_to_minus_safe_max_through_plus_safe_max(helm, amount, factor):
    n = helm.nodes
    h = H.Helm((_axis(safe_max=0.4),))
    h = n.HelmShape().run(h, amount)[0]
    out = n.HelmApplyNode().run(_model(), h)[0]
    res = _run(out, torch.zeros(1, DIM))
    assert torch.allclose(res, factor * 0.4 * _vec().reshape(1, DIM), atol=1e-6)


def test_positive_and_negative_are_symmetric_around_the_untouched_output(helm):
    n = helm.nodes
    x = torch.randn(1, DIM)
    v = torch.randn(DIM)
    d = {}
    for amt in (0.6, -0.6):
        h = n.HelmShape().run(H.Helm((_axis(vec=v),)), amt)[0]
        d[amt] = _run(n.HelmApplyNode().run(_model(), h)[0], x) - x
    assert torch.allclose(d[0.6], -d[-0.6], atol=1e-6)


def test_amount_zero_is_a_passthrough_that_does_not_even_hook_the_model(helm):
    n = helm.nodes
    mp = _model()
    h = n.HelmShape().run(H.Helm((_axis(),)), 0.0)[0]
    assert n.HelmApplyNode().run(mp, h)[0] is mp


# --- Shape ---------------------------------------------------------------------------------------
def test_shape_sets_amount_curve_and_stage_on_every_axis_without_touching_the_input(helm):
    n = helm.nodes
    h = H.Helm((_axis("a"), _axis("b", vec=_vec(2))))
    out = n.HelmShape().run(h, -0.3, stage="late steps (finish)", split=0.6, curve="ramp", end_amount=0.9,
                            easing="ease_in")[0]
    for a in out.axes:
        assert (a.amount, a.stage, a.split) == (-0.3, "late", 0.6)
        assert a.curve.kind == "ramp" and a.curve.end == 0.9 and a.curve.easing == "ease_in"
    assert all(a.amount == 0.5 and a.curve is None and a.stage == "all" for a in h.axes)


def test_shape_keyframes_and_validation(helm):
    n = helm.nodes
    h = H.Helm((_axis(),))
    out = n.HelmShape().run(h, 0.9, curve="keyframes", keyframes="0:0.2, 0.5:0.8, 1:0.4")[0]
    assert out.axes[0].curve.keys == ((0.0, 0.2), (0.5, 0.8), (1.0, 0.4)) and out.axes[0].amount == 0.2
    for bad in ("0:0.2", "0.5:0.2, 0.2:0.5", "0:3, 1:0"):
        with pytest.raises(H.HelmValueError):
            n.HelmShape().run(h, 0.5, curve="keyframes", keyframes=bad)
    with pytest.raises(H.HelmValueError):
        n.HelmShape().run(h, 1.5)
    with pytest.raises(ValueError):
        n.HelmShape().run(h, 0.5, stage="sometimes")


def test_shape_is_flagged_experimental_and_curve_tooltips_say_so(helm):
    n = helm.nodes
    assert n.HelmShape.EXPERIMENTAL is True
    opt = n.HelmShape.INPUT_TYPES()["optional"]
    assert "Experimental" in opt["curve"][1]["tooltip"] and "MiniMax H3" in opt["curve"][1]["tooltip"]
    assert "Experimental" in opt["stage"][1]["tooltip"]


# --- stage on the hook path -----------------------------------------------------------------------
def _stage_hits(helm, stage, split=0.5):
    n = helm.nodes
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    h = n.HelmShape().run(H.Helm((_axis(safe_max=1.0),)), 1.0, stage=stage, split=split)[0]
    out = n.HelmApplyNode().run(_model(), h)[0]
    hits = []
    for s in sigmas[:-1]:
        res = _run(out, torch.zeros(1, DIM), sig=sigmas, sigma=s)
        hits.append(bool(res.abs().sum() > 0))
    return hits


def test_early_stage_acts_only_in_the_first_half_of_the_schedule(helm):
    assert _stage_hits(helm, "early steps (composition)") == [True, True, False, False]


def test_late_stage_acts_only_in_the_second_half(helm):
    assert _stage_hits(helm, "late steps (finish)") == [False, False, True, True]


def test_all_stage_acts_everywhere_and_split_moves_the_boundary(helm):
    assert _stage_hits(helm, "all steps") == [True] * 4
    assert _stage_hits(helm, "early steps (composition)", split=0.2) == [True, False, False, False]
    assert _stage_hits(helm, "late steps (finish)", split=0.7) == [False, False, False, True]


def test_stage_follows_the_sigma_not_the_step_count(helm):
    n = helm.nodes
    sigmas = [1.0, 0.9, 0.8, 0.1, 0.0]                 # a shifted schedule: sigma stays high for 2 of 4 steps
    h = n.HelmShape().run(H.Helm((_axis(safe_max=1.0),)), 1.0, stage="late steps (finish)")[0]
    out = n.HelmApplyNode().run(_model(), h)[0]
    hits = [bool(_run(out, torch.zeros(1, DIM), sig=sigmas, sigma=s).abs().sum() > 0) for s in sigmas[:-1]]
    assert hits == [False, False, False, True]         # progress 0, .11, .22, .89 -> only the last is late


def test_two_axes_can_use_different_stages(helm):
    n = helm.nodes
    e = n.HelmShape().run(H.Helm((_axis("e", vec=_vec(1), safe_max=1.0),)), 1.0, stage="early steps (composition)")[0]
    late = n.HelmShape().run(H.Helm((_axis("l", vec=_vec(2), safe_max=1.0),)), 1.0, stage="late steps (finish)")[0]
    mixed, _ = n.HelmMix().run(e, orthogonalize=False, helm_2=late)
    out = n.HelmApplyNode().run(_model(), mixed)[0]
    sig = [1.0, 0.5, 0.0]
    first = _run(out, torch.zeros(1, DIM), sig=sig, sigma=1.0)
    second = _run(out, torch.zeros(1, DIM), sig=sig, sigma=0.5)
    assert torch.allclose(first, _vec(1).reshape(1, DIM)) and torch.allclose(second, _vec(2).reshape(1, DIM))


# --- time curve on a synthetic H3 layout -----------------------------------------------------------
class MiniMaxH3Model(conftest.FakeDiT):
    """Named like the real class so Apply takes the H3 path."""


H3DIM = 5376   # the calibrated H3 width; Apply checks the axis width against it


def _h3_model(seq=20, dim=H3DIM):
    return conftest.fake_model(MiniMaxH3Model(dim=dim, n=6)), seq


H3_SHAPES = [torch.Size([1, 24, 4, 4, 4]), torch.Size([1, 32, 2, 7])]    # 4 frames, 4 rows per frame = 16 video rows


def test_ramp_scales_each_frames_rows_and_leaves_the_other_rows_alone(helm):
    n = helm.nodes
    mp, seq = _h3_model(20)
    h = n.HelmShape().run(H.Helm((_axis(vec=torch.ones(H3DIM), model="MiniMaxH3Model", safe_max=0.5),)), 0.0,
                          curve="ramp", end_amount=1.0)[0]
    out = n.HelmApplyNode().run(mp, h)[0]
    res = _run(out, torch.zeros(seq, H3DIM), shapes=H3_SHAPES)
    frame_scale = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    expect = torch.cat([torch.zeros(4), frame_scale.repeat_interleave(4)]) * 0.5
    assert torch.allclose(res[:, 0], expect, atol=1e-6)
    assert torch.allclose(res, expect.unsqueeze(1).expand(seq, H3DIM), atol=1e-6)


def test_keyframes_curve_with_a_negative_segment(helm):
    n = helm.nodes
    mp, seq = _h3_model(20)
    h = n.HelmShape().run(H.Helm((_axis(vec=torch.ones(H3DIM), model="MiniMaxH3Model", safe_max=1.0),)), 0.0,
                          curve="keyframes", keyframes="0:1, 1:-1")[0]
    res = _run(n.HelmApplyNode().run(mp, h)[0], torch.zeros(seq, H3DIM), shapes=H3_SHAPES)
    per_frame = res[4:, 0].reshape(4, 4)[:, 0]
    assert torch.allclose(per_frame, torch.tensor([1.0, 1 / 3, -1 / 3, -1.0]), atol=1e-6)


def test_curve_on_a_model_without_a_known_video_layout_uses_the_average_and_says_so(helm, capsys):
    n = helm.nodes
    h = n.HelmShape().run(H.Helm((_axis(vec=torch.ones(DIM), safe_max=1.0),)), 0.0, curve="ramp", end_amount=1.0)[0]
    res = _run(n.HelmApplyNode().run(_model(), h)[0], torch.zeros(1, DIM))
    assert torch.allclose(res, torch.full((1, DIM), 0.5), atol=0.01)
    assert "only available for MiniMax H3" in capsys.readouterr().out


def test_curve_refuses_a_layout_that_does_not_fit(helm):
    n = helm.nodes
    mp, _ = _h3_model(10)
    h = n.HelmShape().run(H.Helm((_axis(vec=torch.ones(H3DIM), model="MiniMaxH3Model"),)), 0.0, curve="ramp", end_amount=1.0)[0]
    out = n.HelmApplyNode().run(mp, h)[0]
    with pytest.raises(RuntimeError, match="refused rather than guessed"):
        _run(out, torch.zeros(10, H3DIM), shapes=H3_SHAPES)          # 16 video rows do not fit in 10


def test_constant_amount_on_h3_applies_to_every_row_like_the_dial(helm):
    n = helm.nodes
    mp, seq = _h3_model(20)
    h = n.HelmShape().run(H.Helm((_axis(vec=torch.ones(H3DIM), model="MiniMaxH3Model", safe_max=1.0),)), 0.5)[0]
    res = _run(n.HelmApplyNode().run(mp, h)[0], torch.zeros(seq, H3DIM), shapes=H3_SHAPES)
    assert torch.allclose(res, torch.full((seq, H3DIM), 0.5))


# --- Mix -----------------------------------------------------------------------------------------
def test_mix_combines_inputs_and_applies_input_and_per_axis_weights(helm):
    n = helm.nodes
    a = H.Helm((_axis("a", vec=_vec(1)),))
    b = H.Helm((_axis("b", vec=_vec(2)), _axis("c", vec=_vec(3))))
    out, report = n.HelmMix().run(a, orthogonalize=False, weight_1=0.5, helm_2=b, weight_2=1.0,
                                  axis_weights="1, 0.5, 1")
    assert [x.label for x in out.axes] == ["a", "b", "c"]
    assert [x.weight for x in out.axes] == pytest.approx([0.5, 0.5, 1.0])
    assert isinstance(report, str) and "3 idea" in report
    assert all(x.weight == 1.0 for x in a.axes + b.axes)            # inputs untouched


def test_mix_orthogonalizes_by_default_and_moving_one_idea_does_not_move_the_other(helm):
    n = helm.nodes
    a = H.Helm((_axis("a", vec=torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])),))
    b = H.Helm((_axis("b", vec=torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0])),))
    mixed, _ = n.HelmMix().run(a, helm_2=b)
    va, vb = (x.vec for x in mixed.axes)
    assert abs(float(va @ vb)) < 1e-6
    ua = va / va.norm()
    along_a = []
    for wb in (0.0 + 1e-9, 0.3, 1.0):                  # sweep the second idea
        m, _ = n.HelmMix().run(a, helm_2=b, weight_2=wb)
        res = _run(n.HelmApplyNode().run(_model(), m)[0], torch.zeros(1, DIM))
        along_a.append(float(res[0] @ ua))
    assert max(along_a) - min(along_a) < 1e-5          # the component along the first idea never moved
    plain, _ = n.HelmMix().run(a, orthogonalize=False, helm_2=b)
    assert abs(float(plain.axes[0].vec @ plain.axes[1].vec)) > 0.5


def test_mix_warns_about_models_and_blocks_and_carries_the_warning_to_apply(helm, capsys):
    n = helm.nodes
    a = H.Helm((_axis("a", vec=_vec(1), block=2),))
    b = H.Helm((_axis("b", vec=_vec(2), block=4),))
    out, report = n.HelmMix().run(a, helm_2=b)
    assert "different blocks" in report and out.warnings
    n.HelmApplyNode().run(_model(), out)
    assert "different blocks" in capsys.readouterr().out
    c = H.Helm((_axis("c", vec=_vec(3), model="Other"),))
    _, report2 = n.HelmMix().run(a, helm_2=c)
    assert "different models" in report2


def test_mix_validates_its_inputs(helm):
    n = helm.nodes
    a = H.Helm((_axis(),))
    with pytest.raises(H.HelmValueError):
        n.HelmMix().run(a, axis_weights="1, 1")
    with pytest.raises(H.HelmValueError):
        n.HelmMix().run(a, helm_2="nope")
    with pytest.raises(H.HelmValueError):
        n.HelmMix().run(a, weight_1=1.5)


def test_mix_of_a_chain_counts_each_axis(helm):
    n = helm.nodes
    chain = H.Helm((_axis("a", vec=_vec(1)), _axis("b", vec=_vec(2)), _axis("c", vec=_vec(3))))
    out, _ = n.HelmMix().run(chain, axis_weights="1, 0, 1")
    assert [x.weight for x in out.axes] == [1.0, 0.0, 1.0]


# --- Apply ---------------------------------------------------------------------------------------
def test_apply_applies_a_chain_as_is_each_at_its_own_block(helm):
    n = helm.nodes
    h = H.Helm((_axis("a", vec=_vec(1), block=1, safe_max=1.0), _axis("b", vec=_vec(2), block=3, safe_max=1.0)))
    out = n.HelmApplyNode().run(_model(), h)[0]
    assert len(out.model_options["helm_hooks"]) == 2
    res = _run(out, torch.zeros(1, DIM))
    assert torch.allclose(res, 0.5 * (_vec(1) + _vec(2)).reshape(1, DIM))


def test_apply_weight_scales_the_strength(helm):
    n = helm.nodes
    h = H.Helm((_axis(vec=torch.ones(DIM), safe_max=0.4, weight=0.5, amount=1.0),))
    res = _run(n.HelmApplyNode().run(_model(), h)[0], torch.zeros(1, DIM))
    assert torch.allclose(res, torch.full((1, DIM), 0.2))


def test_apply_refuses_the_wrong_model_and_missing_ranges_in_plain_words(helm):
    n = helm.nodes
    with pytest.raises(ValueError, match="made for"):
        n.HelmApplyNode().run(_model(), H.Helm((_axis(model="NextDiT"),)))
    with pytest.raises(ValueError, match="no safe range"):
        n.HelmApplyNode().run(_model(), H.Helm((_axis(safe_max=None),)))
    with pytest.raises(ValueError, match="which block"):
        n.HelmApplyNode().run(_model(), H.Helm((_axis(block=None),)))
    with pytest.raises(H.HelmValueError):
        n.HelmApplyNode().run(_model(), "not a helm")


def test_apply_composes_with_an_existing_lab_apply(helm):
    n = helm.nodes
    _save_library_axis(helm, "lab", safe_max=0.4)
    mp = n.HelmApply().run(_model(), "lab", 2, 0.1)[0]
    h = H.Helm((_axis(vec=torch.ones(DIM), safe_max=1.0),))
    out = n.HelmApplyNode().run(mp, h)[0]
    assert len(out.model_options["helm_hooks"]) == 2
    base = torch.arange(1, DIM + 1, dtype=torch.float32)
    assert torch.allclose(_run(out, torch.zeros(1, DIM)), (0.1 * base + 0.5).reshape(1, DIM))


def test_apply_does_not_modify_the_helm_or_the_original_model(helm):
    n = helm.nodes
    mp = _model()
    h = H.Helm((_axis(),))
    before = h.to_dict()
    n.HelmApplyNode().run(mp, h)
    assert h.to_dict() == before and not mp.model_options


def test_cond_only_flag_passes_through_the_same_hook_machinery(helm):
    n = helm.nodes
    out = n.HelmApplyNode().run(_model(), H.Helm((_axis(),)), cond_only=True)[0]
    assert out.model_options["helm_hooks"][0]["cond_only"] is True


# --- Axis from prompts: multi-pair, debias ------------------------------------------------------
PDIM = 64


def _unit(seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(PDIM, generator=g)
    return v / v.norm()


IDEA, TINT = _unit(1), _unit(2)
IDEA = IDEA - (IDEA @ TINT) * TINT
IDEA = IDEA / IDEA.norm()


def _prompt_world(helm):
    """text -> mean activation. Every 'with idea' text carries IDEA plus a shared TINT (the nuisance), plus
    its own subject; nuisance pairs differ along TINT and a little noise."""
    n = helm.nodes
    world, pairs = {}, []
    for k in range(4):
        subj = _unit(100 + k) * 3
        a, b = f"pair {k} with", f"pair {k} without"
        world[a] = subj + 2.0 * IDEA + 1.5 * TINT
        world[b] = subj
        pairs.append(f"{a} || {b}")
    for k, (a, b) in enumerate(n.NUISANCE_PAIRS):
        subj = _unit(300 + k) * 3
        c = 1.0 + 0.3 * k
        world[a] = subj + c * TINT + 0.01 * _unit(400 + k)
        world[b] = subj
    return world, "\n".join(pairs)


class Recorder(torch.nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def forward(self, x):
        return self.store["act"]


def _prompt_model():
    store = {"act": torch.zeros(1, 1, PDIM)}
    dit = make_dit(dim=PDIM, n=6)
    dit.blocks[2] = Recorder(store)
    return conftest.fake_model(dit), dit, store


def _drive_for(world, dit, store):
    def drive(mp, text):
        store["act"] = world[text].reshape(1, 1, PDIM)
        w = mp.model_options["model_function_wrapper"]
        for _ in range(2):
            w(make_apply_model(dit), make_c_in(torch.zeros(1, PDIM), sig=SIG))
    return drive


def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def test_axis_from_several_pairs_averages_and_saves_a_self_describing_file(helm):
    n = helm.nodes
    world, pairs = _prompt_world(helm)
    mp, dit, store = _prompt_model()
    (h,) = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="multi", debias=False, block=2,
                            _drive=_drive_for(world, dit, store))
    a = h.axes[0]
    assert a.source == "pairs" and a.info["n_pairs"] == 4 and a.info["debiased"] is False
    expected = torch.stack([world[f"pair {k} with"] - world[f"pair {k} without"] for k in range(4)]).mean(0)
    assert torch.allclose(a.vec, expected, atol=1e-5)
    saved = n._load_axis(n._resolve_axis("multi"))
    assert torch.allclose(saved, a.vec) and (n._read_meta("multi") or {}).get("made_by") == "HelmAxis"
    assert a.safe_max > 0 and "estimate" in a.info["safe_max_source"]
    assert any(e["file"] == "multi" for e in n._dial_entries())      # shows up in the library under Custom


def test_axis_debias_removes_the_shared_tint_that_a_plain_pair_axis_keeps(helm):
    n = helm.nodes
    world, pairs = _prompt_world(helm)
    mp, dit, store = _prompt_model()
    drive = _drive_for(world, dit, store)
    raw = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="raw", debias=False, block=2, _drive=drive)[0].axes[0]
    deb = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="deb", debias=True, block=2, _drive=drive)[0].axes[0]
    assert deb.info["debiased"] is True and deb.info["nuisance_rank"] >= 1
    assert abs(_cos(deb.vec, TINT)) < 0.05 < abs(_cos(raw.vec, TINT))
    assert _cos(deb.vec, IDEA) > _cos(raw.vec, IDEA)
    assert deb.info["nuisance_removed"] > 0.2


def test_debias_strength_zero_equals_no_debias_and_nuisance_is_cached(helm):
    n = helm.nodes
    world, pairs = _prompt_world(helm)
    mp, dit, store = _prompt_model()
    calls = {"n": 0}
    base = _drive_for(world, dit, store)

    def drive(m, t):
        calls["n"] += 1
        base(m, t)

    raw = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="r", debias=False, block=2, _drive=drive)[0].axes[0]
    zero = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="z", debias=True, debias_strength=0.0, block=2,
                            _drive=drive)[0].axes[0]
    assert torch.allclose(raw.vec, zero.vec)
    first = calls["n"]
    n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="d1", debias=True, block=2, _drive=drive)
    after_first_debias = calls["n"]
    n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="d2", debias=True, block=2, _drive=drive)
    pair_passes = 2 * 4
    assert after_first_debias - first >= 2 * len(n.NUISANCE_PAIRS) + pair_passes
    assert calls["n"] - after_first_debias == pair_passes                # nuisance not recomputed


def test_axis_from_prompts_validates_inputs(helm):
    n = helm.nodes
    mp, dit, store = _prompt_model()
    with pytest.raises(ValueError, match="needs the model"):
        n.HelmAxis().run("prompts", "x", pairs="a || b")
    with pytest.raises(H.HelmValueError):
        n.HelmAxis().run("prompts", "x", model=mp, pairs="no separator here", block=2, _drive=lambda m, t: None)
    with pytest.raises(ValueError, match="at least one pair"):
        n.HelmAxis().run("prompts", "x", model=mp, pairs="# nothing", block=2, _drive=lambda m, t: None)
    with pytest.raises(ValueError, match="does not exist"):
        n.HelmAxis().run("prompts", "x", model=mp, pairs="a || b", block=99, _drive=lambda m, t: None)


def test_axis_from_prompts_chains_and_applies_end_to_end(helm):
    n = helm.nodes
    world, pairs = _prompt_world(helm)
    mp, dit, store = _prompt_model()
    h = n.HelmAxis().run("prompts", "x", model=mp, pairs=pairs, name="e2e", debias=False, block=2,
                         _drive=_drive_for(world, dit, store))[0]
    h = n.HelmShape().run(h, -0.5)[0]
    out = n.HelmApplyNode().run(mp, h)[0]
    a = h.axes[0]
    store["act"] = torch.zeros(1, 1, PDIM)               # what block 2 would output; Helm adds the idea on top
    res = _run(out, torch.zeros(1, PDIM))
    assert torch.allclose(res[0, 0], -0.5 * a.safe_max * a.vec, atol=1e-6)


def test_nuisance_pairs_are_plain_english_and_not_empty(helm):
    n = helm.nodes
    assert len(n.NUISANCE_PAIRS) >= 6
    for a, b in n.NUISANCE_PAIRS:
        assert a != b and a.strip() and b.strip()


# --- backward compatibility ------------------------------------------------------------------------
LAB = ("HelmCollect", "HelmVector", "HelmApply", "HelmProbe", "HelmProbeReport")


def test_old_classes_are_all_still_registered_with_lab_category(helm):
    n = helm.nodes
    for cls in LAB + ("HelmDial", "HelmMakeAxis"):
        assert cls in n.NODE_CLASS_MAPPINGS and cls in n.NODE_DISPLAY_NAME_MAPPINGS
    for cls in LAB:
        assert n.NODE_CLASS_MAPPINGS[cls].CATEGORY == "Helm/Lab"


def test_lab_nodes_are_hidden_from_search_with_comfyuis_deprecated_flag(helm):
    n = helm.nodes
    for cls in LAB:
        assert n.NODE_CLASS_MAPPINGS[cls].DEPRECATED is True, cls
    for cls in ("HelmAxis", "HelmShape", "HelmMix", "HelmApplyNode", "HelmDial", "HelmMakeAxis"):
        assert not getattr(n.NODE_CLASS_MAPPINGS[cls], "DEPRECATED", False), cls


def test_lab_inputs_are_unchanged_so_saved_widget_values_still_line_up(helm):
    req = list(helm.nodes.HelmApply.INPUT_TYPES()["required"])
    assert req == ["model", "axis", "block", "strength", "dial", "cond_only", "start_percent", "end_percent"]
    opt = list(helm.nodes.HelmApply.INPUT_TYPES()["optional"])
    assert opt == ["mode", "mask", "layout", "grid", "clamp_k", "mask_offset"]


def test_old_workflow_files_only_use_node_types_that_still_exist(helm):
    files = glob.glob(os.path.join(helm.pack, "example_workflows", "**", "*.json"), recursive=True)
    files += glob.glob(os.path.join(helm.pack, "api", "*.json"))
    assert files
    seen = set()
    for path in files:
        data = json.load(open(path, encoding="utf-8"))
        nodes = data["nodes"] if "nodes" in data else list(data.values())
        for node in nodes:
            t = node.get("type") or node.get("class_type")
            if isinstance(t, str) and t.startswith("Helm"):
                seen.add(t)
                assert t in helm.nodes.NODE_CLASS_MAPPINGS, (path, t)
    assert "HelmApply" in seen


def test_an_old_helm_apply_graph_still_runs_end_to_end(helm):
    n = helm.nodes
    _save_library_axis(helm, "old", safe_max=0.4)
    out = n.HelmApply().run(_model(), "old", 2, 0.1, 0.0, False, 0.0, 1.0)[0]
    base = torch.arange(1, DIM + 1, dtype=torch.float32).reshape(1, DIM)
    assert torch.allclose(_run(out, torch.zeros(1, DIM)), 0.1 * base)


def test_the_wrapper_still_passes_ctx_only_to_hooks_that_ask_for_it(helm):
    n = helm.nodes
    seen = {}

    def lab_fn(m, args, out, flags=None):
        seen["lab"] = True

    def new_fn(m, args, out, flags=None, ctx=None):
        seen["ctx"] = ctx

    mp = _model()
    n._add_hook(mp, {"block": 1, "fn": lab_fn, "start": 0.0, "end": 1.0})
    n._add_hook(mp, {"block": 2, "fn": new_fn, "start": 0.0, "end": 1.0, "ctx": True})
    n._install_wrapper(mp)
    _run(mp, torch.zeros(1, DIM), sig=SIG, sigma=1.0, shapes=[torch.Size([1, 2, 3, 4, 5])])
    assert seen["lab"] and seen["ctx"]["progress"] == 0.0 and seen["ctx"]["latent_shapes"][0][2] == 3
