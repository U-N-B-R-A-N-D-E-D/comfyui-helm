"""Pure maths: curves, easing, stages, the H3 token layout, averaging, debias, Gram-Schmidt."""
import pytest
import torch

import helm_core as H


# --- curve and easing ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", H.EASINGS)
def test_every_easing_starts_at_0_and_ends_at_1(name):
    assert float(H.ease(name, torch.tensor(0.0))) == 0.0 and float(H.ease(name, torch.tensor(1.0))) == 1.0


def test_easing_shapes():
    t = torch.tensor(0.5)
    assert float(H.ease("ease_in", t)) < 0.5 < float(H.ease("ease_out", t))
    assert float(H.ease("ease_in_out", t)) == pytest.approx(0.5)
    with pytest.raises(H.HelmValueError):
        H.ease("bounce", t)


def test_constant_curve_is_flat():
    s = H.curve_series(0.3, None, torch.linspace(0, 1, 5))
    assert torch.allclose(s, torch.full((5,), 0.3))
    assert torch.allclose(H.curve_series(-0.7, H.HelmCurve(), torch.linspace(0, 1, 5)), torch.full((5,), -0.7))


def test_ramp_runs_from_amount_to_end_and_can_cross_zero():
    s = H.curve_series(-1.0, H.HelmCurve("ramp", end=1.0), torch.linspace(0, 1, 5))
    assert torch.allclose(s, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    e = H.curve_series(0.0, H.HelmCurve("ramp", end=1.0, easing="ease_in"), torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(e, torch.tensor([0.0, 0.25, 1.0]))


def test_keyframes_interpolate_and_hold_outside():
    c = H.HelmCurve("keys", keys=((0.25, 0.0), (0.75, 1.0)))
    s = H.curve_series(0.0, c, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert torch.allclose(s, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_parse_keyframes():
    assert H.parse_keyframes("0:0.2, 0.5:0.8; 1:-0.4") == ((0.0, 0.2), (0.5, 0.8), (1.0, -0.4))
    for bad in ("0.5", "a:b", "0:0.2:0.4"):
        with pytest.raises(H.HelmValueError):
            H.parse_keyframes(bad)


def test_frame_times():
    assert torch.allclose(H.frame_times(5), torch.tensor([0, .25, .5, .75, 1.0]))
    assert torch.equal(H.frame_times(1), torch.zeros(1))
    with pytest.raises(H.HelmValueError):
        H.frame_times(0)


# --- stage ---------------------------------------------------------------------------------------
def test_progress_uses_the_sigma_against_the_schedule():
    sig = [1.0, 0.8, 0.4, 0.1, 0.0]
    assert H.progress_from_sigma(1.0, sig, 0, 4) == pytest.approx(0.0)
    assert H.progress_from_sigma(0.4, sig, 2, 4) == pytest.approx(0.6)   # sigma, not the step count (0.5)
    assert H.progress_from_sigma(0.0, sig, 4, 4) == pytest.approx(1.0)


def test_progress_falls_back_to_steps_when_sigma_is_unusable():
    sig = [1.0, 0.5, 0.0]
    assert H.progress_from_sigma(None, sig, 1, 2) == 0.5
    assert H.progress_from_sigma(999.0, sig, 1, 2) == 0.5           # a timestep in other units never mis-gates
    assert H.progress_from_sigma(0.3, None, 3, 4) == 0.75
    assert H.progress_from_sigma(None, None, 0, None) == 0.0


def test_stage_windows_partition_the_render():
    early, late, allw = (H.stage_window(s, 0.4) for s in ("early", "late", "all"))
    for p in (0.0, 0.2, 0.39, 0.4, 0.7, 0.99, 1.0):
        assert H.stage_active(early, p) != H.stage_active(late, p), p      # exactly one of the two
        assert H.stage_active(allw, p)
    assert H.stage_active(early, 0.39) and not H.stage_active(early, 0.4)
    with pytest.raises(H.HelmValueError):
        H.stage_window("middle")
    with pytest.raises(H.HelmValueError):
        H.stage_window("early", 1.0)


# --- H3 layout ----------------------------------------------------------------------------------
def test_h3_layout_video_is_the_last_t_times_half_h_half_w_rows():
    first, t, per = H.h3_video_layout(100, [1, 24, 5, 6, 8])
    assert (t, per) == (5, 12) and first == 100 - 60


def test_h3_layout_pads_odd_latents_to_the_patch_like_the_core_does():
    _, t, per = H.h3_video_layout(100, [1, 24, 3, 5, 7])     # 5x7 pads to 6x8 -> 3*4 rows per frame
    assert per == 12 and t == 3


def test_h3_layout_refuses_what_does_not_fit():
    with pytest.raises(H.HelmValueError):
        H.h3_video_layout(10, [1, 24, 5, 6, 8])
    with pytest.raises(H.HelmValueError):
        H.h3_video_layout(100, [24, 5, 6, 8])


def test_frame_row_scale_is_frame_major():
    layout = H.h3_video_layout(10, [1, 24, 3, 2, 2])         # 3 frames x 1 row each... (2/2)*(2/2)=1
    assert layout == (7, 3, 1)
    s = H.frame_row_scale(10, layout, [0.1, 0.2, 0.3], outside=0.0)
    assert torch.allclose(s, torch.tensor([0, 0, 0, 0, 0, 0, 0, .1, .2, .3]))
    layout = (2, 2, 3)
    s = H.frame_row_scale(8, layout, [1.0, -1.0], outside=0.5)
    assert torch.allclose(s, torch.tensor([.5, .5, 1, 1, 1, -1, -1, -1]))
    with pytest.raises(H.HelmValueError):
        H.frame_row_scale(8, layout, [1.0])


# --- multi-pair averaging ---------------------------------------------------------------------
def _unit(dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_averaging_pairs_cancels_subject_specific_detail():
    dim, idea = 256, _unit(256, 0)
    pos, neg = [], []
    for k in range(12):
        subject_detail = 0.8 * _unit(dim, 100 + k)            # differs per pair, same on both sides? no: leaks into the diff
        base = _unit(dim, 200 + k)
        pos.append(base + idea + subject_detail)
        neg.append(base - subject_detail * 0.2)
    diffs = H.pair_diffs(pos, neg)
    one, avg = diffs[0], H.average_pairs(diffs)
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    assert cos(avg, idea) > cos(one, idea) + 0.15
    assert all(c > 0 for c in H.pair_agreement(diffs))


def test_pair_diffs_validates():
    with pytest.raises(H.HelmValueError):
        H.pair_diffs([torch.ones(3)], [])
    with pytest.raises(H.HelmValueError):
        H.pair_diffs([torch.ones(3)], [torch.ones(4)])


def test_parse_pairs():
    text = "# comment\n\nfoggy street || clear street\n  a  ||  b  \n"
    assert H.parse_pairs(text) == [("foggy street", "clear street"), ("a", "b")]
    for bad in ("no separator", "left ||", "|| right", "same || same"):
        with pytest.raises(H.HelmValueError):
            H.parse_pairs(bad)


# --- debias -------------------------------------------------------------------------------------
def test_nuisance_basis_is_orthonormal_and_spans_the_nuisance_directions():
    dim = 64
    n1, n2 = _unit(dim, 1), _unit(dim, 2)
    diffs = torch.stack([3 * n1, -2 * n1 + 0.5 * n2, n2, 0.1 * n1])
    q = H.nuisance_basis(diffs)
    assert q.shape[0] == 2
    assert torch.allclose(q @ q.T, torch.eye(2), atol=1e-5)
    for n in (n1, n2):
        assert float((n @ q.T).norm()) == pytest.approx(1.0, abs=1e-4)      # n lies inside the span


def test_debias_removes_the_shared_component_and_keeps_the_idea():
    dim = 128
    idea, tint = _unit(dim, 10), _unit(dim, 11)
    idea = idea - (idea @ tint) * tint
    idea = idea / idea.norm()
    raw = 2.0 * idea + 1.5 * tint                               # an axis read as a tint
    q = H.nuisance_basis(torch.stack([tint, -tint * 2, tint * 0.5]))
    out, removed = H.debias_project(raw, q)
    assert abs(float(out @ tint)) < 1e-4
    assert float(out @ idea) == pytest.approx(2.0, abs=1e-4)    # the idea is untouched
    assert removed == pytest.approx(1.5 ** 2 / (2.0 ** 2 + 1.5 ** 2), abs=1e-4)


def test_debias_strength_is_a_partial_projection_and_zero_is_identity():
    dim = 32
    tint = _unit(dim, 5)
    v = 2 * _unit(dim, 6) + tint
    q = H.nuisance_basis(tint.unsqueeze(0))
    full, _ = H.debias_project(v, q, 1.0)
    half, _ = H.debias_project(v, q, 0.5)
    none, rem = H.debias_project(v, q, 0.0)
    assert torch.equal(none, v) and rem == 0.0
    assert abs(float(half @ tint)) == pytest.approx(0.5 * abs(float(v @ tint)), rel=1e-4)
    assert abs(float(full @ tint)) < 1e-5
    with pytest.raises(H.HelmValueError):
        H.debias_project(v, q, 1.5)


def test_debias_reports_an_axis_that_is_all_nuisance():
    tint = _unit(16, 3)
    q = H.nuisance_basis(tint.unsqueeze(0))
    out, removed = H.debias_project(5 * tint, q)
    assert removed == pytest.approx(1.0, abs=1e-5) and float(out.norm()) < 1e-4


def test_nuisance_basis_refuses_no_direction():
    with pytest.raises(H.HelmValueError):
        H.nuisance_basis(torch.zeros(3, 8))


# --- Gram-Schmidt -----------------------------------------------------------------------------
def test_gram_schmidt_makes_the_vectors_orthogonal_and_keeps_the_first():
    vs = [_unit(64, i) * (1 + i) for i in range(4)]
    out, dropped = H.gram_schmidt(vs)
    assert dropped == [] and torch.equal(out[0], vs[0])
    for i in range(4):
        for j in range(i):
            assert abs(float(out[i] @ out[j])) < 1e-4 * float(out[i].norm() * out[j].norm()) + 1e-5
    for o, v in zip(out, vs):                                    # norms preserved: amounts keep their meaning
        assert float(o.norm()) == pytest.approx(float(v.norm()), rel=1e-5)


def test_gram_schmidt_without_norm_preservation_and_independence_of_movement():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 1.0, 0.0])
    (_, ob), _ = H.gram_schmidt([a, b], preserve_norm=False)
    assert torch.allclose(ob, torch.tensor([0.0, 1.0, 0.0]))
    # moving b's axis no longer changes the projection on a
    assert float(ob @ a) == 0.0


def test_gram_schmidt_drops_a_dependent_vector():
    a = _unit(16, 1)
    out, dropped = H.gram_schmidt([a, 3 * a, _unit(16, 2)])
    assert dropped == [1] and out[1] is None and out[2] is not None
    assert abs(float(out[2] @ a)) < 1e-5


def test_gram_schmidt_ignores_nothing_it_is_not_asked_to():
    out, dropped = H.gram_schmidt([_unit(8, 1)])
    assert dropped == [] and len(out) == 1


# --- weights / mix ----------------------------------------------------------------------------
def test_parse_weights():
    assert H.parse_weights("", 3) == [1.0, 1.0, 1.0]
    assert H.parse_weights("1, 0.5; 0", 3) == [1.0, 0.5, 0.0]
    for bad in ("1, 2", "1, 0.5, 2", "a, b, c"):
        with pytest.raises(H.HelmValueError):
            H.parse_weights(bad, 3)


def _ax(name, vec, **kw):
    return H.HelmAxis(name=name, label=name, vec=torch.tensor(vec, dtype=torch.float32), model="M", block=2,
                      safe_max=1.0, **kw)


def test_mix_axes_orthogonalizes_only_active_axes_of_the_same_model_and_block():
    a, b = _ax("a", [1, 0, 0, 0]), _ax("b", [1, 1, 0, 0])
    c = _ax("c", [1, 1, 1, 0], weight=0.0)                      # inactive: never influences the others
    out, warns = H.mix_axes([a, b, c])
    assert warns == []
    assert torch.allclose(out[1].vec, torch.tensor([0.0, 1.41421356, 0, 0]), atol=1e-5)
    assert torch.equal(out[2].vec, c.vec) and out[1].info.get("orthogonalized") is True
    assert abs(float(out[0].vec @ out[1].vec)) < 1e-6


def test_mix_axes_can_leave_vectors_alone():
    a, b = _ax("a", [1, 0, 0, 0]), _ax("b", [1, 1, 0, 0])
    out, _ = H.mix_axes([a, b], orthogonalize=False)
    assert torch.equal(out[1].vec, b.vec)


def test_mix_axes_warns_on_different_models_and_blocks():
    a = _ax("a", [1, 0, 0, 0])
    other_model = H.HelmAxis(name="m", label="m", vec=torch.ones(4), model="N", block=2, safe_max=1.0)
    other_block = H.HelmAxis(name="k", label="k", vec=torch.ones(4), model="M", block=5, safe_max=1.0)
    _, w1 = H.mix_axes([a, other_model])
    _, w2 = H.mix_axes([a, other_block])
    assert any("different models" in w for w in w1)
    assert any("different blocks" in w for w in w2)


def test_mix_axes_reports_a_dependent_axis():
    a, b = _ax("a", [1, 0, 0, 0]), _ax("b", [2, 0, 0, 0])
    out, warns = H.mix_axes([a, b])
    assert any("nothing independent left" in w for w in warns) and torch.equal(out[1].vec, b.vec)
