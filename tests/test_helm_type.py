"""The HELM type: strict validation, immutability, JSON round trip."""
import copy
import json

import pytest
import torch

import helm_core as H


def axis(**kw):
    base = dict(name="a", label="An idea", vec=torch.arange(1, 9, dtype=torch.float32), model="FakeDiT", block=2,
                safe_max=0.4)
    base.update(kw)
    return H.HelmAxis(**base)


def test_a_valid_axis_has_the_documented_defaults():
    a = axis()
    assert (a.amount, a.weight, a.stage, a.curve, a.source) == (0.5, 1.0, "all", None, "other")
    assert a.dim == 8 and a.vec.dtype == torch.float32 and a.is_active and not a.has_curve


@pytest.mark.parametrize("kw", [
    {"vec": torch.zeros(8)},                                  # moves nothing
    {"vec": torch.tensor([1.0, float("nan")])},               # NaN
    {"vec": torch.tensor([1.0, float("inf")])},               # inf
    {"vec": torch.ones(2, 3)},                                # not 1-D
    {"vec": [1.0, 2.0]},                                      # not a tensor
    {"amount": 1.5}, {"amount": -1.01}, {"amount": float("nan")}, {"amount": True}, {"amount": "0.5"},
    {"weight": -0.1}, {"weight": 1.1},
    {"stage": "middle"}, {"split": 0.0}, {"split": 1.0},
    {"safe_max": 0.0}, {"safe_max": -1.0}, {"safe_max": float("inf")},
    {"block": -1}, {"block": 1.5}, {"block": True},
    {"name": ""}, {"label": "   "}, {"name": "x" * 500},
    {"source": "elsewhere"},
    {"curve": "ramp"},
    {"info": {"k": [1, 2]}}, {"info": {"k": float("nan")}}, {"info": {str(i): i for i in range(20)}},
])
def test_invalid_axes_are_refused(kw):
    with pytest.raises(H.HelmValueError):
        axis(**kw)


@pytest.mark.parametrize("kw", [
    {"kind": "spline"}, {"kind": "ramp", "end": 2.0}, {"kind": "ramp", "easing": "bounce"},
    {"kind": "keys", "keys": ((0.0, 0.1),)},                                   # one key
    {"kind": "keys", "keys": ((0.0, 0.1), (0.0, 0.5))},                        # not increasing
    {"kind": "keys", "keys": ((0.0, 0.1), (1.5, 0.5))},                        # position out of range
    {"kind": "keys", "keys": ((0.0, 0.1), (1.0, 3.0))},                        # amount out of range
    {"kind": "keys", "keys": tuple((i / 10, 0.0) for i in range(9))},          # too many
    {"kind": "ramp", "keys": ((0.0, 0.1), (1.0, 0.2))},                        # keys on a ramp
])
def test_invalid_curves_are_refused(kw):
    with pytest.raises(H.HelmValueError):
        H.HelmCurve(**kw)


def test_helm_needs_one_to_sixteen_axes_of_the_right_type():
    with pytest.raises(H.HelmValueError):
        H.Helm(())
    with pytest.raises(H.HelmValueError):
        H.Helm((axis(),) * 17)
    with pytest.raises(H.HelmValueError):
        H.Helm(("not an axis",))
    assert len(H.Helm((axis(),) * 16).axes) == 16


def test_validate_helm_names_the_input():
    with pytest.raises(H.HelmValueError, match="'helm_2' must be a HELM"):
        H.validate_helm("a string", "helm_2")


def test_axis_is_immutable_and_with_changes_copies():
    a = axis()
    with pytest.raises(Exception):
        a.amount = 0.9
    b = a.with_changes(amount=-0.3)
    assert a.amount == 0.5 and b.amount == -0.3 and b.vec is a.vec or torch.equal(b.vec, a.vec)


def test_of_appends_and_never_mutates_the_parent():
    p = H.Helm.of(axis(name="p", label="P"))
    q = H.Helm.of(axis(name="q", label="Q"), p)
    assert [a.label for a in p.axes] == ["P"] and [a.label for a in q.axes] == ["P", "Q"]


def test_json_round_trip_is_lossless_and_json_safe():
    curve = H.HelmCurve("keys", keys=((0.0, 0.2), (0.5, -0.8), (1.0, 0.4)))
    h = H.Helm((axis(amount=-0.25, weight=0.5, curve=curve, stage="late", split=0.6, info={"n_pairs": 3, "ok": True}),
                axis(name="b", label="B", source="pairs")), ("careful",))
    text = json.dumps(h.to_dict())          # must be plain JSON
    back = H.Helm.from_dict(json.loads(text))
    assert back.to_dict() == h.to_dict()
    a0 = back.axes[0]
    assert (a0.amount, a0.weight, a0.stage, a0.split) == (-0.25, 0.5, "late", 0.6)
    assert a0.curve.keys == curve.keys and torch.equal(a0.vec, h.axes[0].vec)
    assert back.warnings == ("careful",)


def test_from_dict_is_strict():
    good = H.Helm((axis(),)).to_dict()
    for mutate in (
        lambda d: d.update(extra=1),
        lambda d: d.pop("warnings"),
        lambda d: d.update(helm_version=99),
        lambda d: d["axes"][0].update(unknown="x"),
        lambda d: d["axes"][0].update(vec=["a", "b"]),
        lambda d: d["axes"][0].update(vec=[True, False]),
        lambda d: d["axes"][0].update(amount=5),
        lambda d: d.update(axes="nope"),
    ):
        d = copy.deepcopy(good)
        mutate(d)
        with pytest.raises(H.HelmValueError):
            H.Helm.from_dict(d)


def test_effective_amount_and_activity():
    assert axis(amount=0.0).is_active is False
    assert axis(weight=0.0).is_active is False
    assert axis(amount=0.0, curve=H.HelmCurve("ramp", end=0.5)).is_active is True
    assert axis(amount=-0.4, weight=0.5).effective_amount() == pytest.approx(-0.2)
    ramp = axis(amount=0.0, curve=H.HelmCurve("ramp", end=1.0))
    assert ramp.mean_amount() == pytest.approx(0.5, abs=0.01)
