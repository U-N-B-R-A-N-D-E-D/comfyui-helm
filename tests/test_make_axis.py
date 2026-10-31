"""HelmMakeAxis: both contrast passes in one execution, same math as HelmCollect + HelmVector."""
import types

import pytest
import torch

import conftest
from conftest import make_apply_model, make_c_in, make_dit

SIG = [1.0, 0.5, 0.0]  # two sampling steps


class Recorder(torch.nn.Module):
    """A block that outputs whatever tensor the current 'prompt' dictates, so the hook reads synthetic data."""

    def __init__(self, store):
        super().__init__()
        self.store = store

    def forward(self, x):
        return self.store["act"]


def _model(dim=8, n=6):
    store = {"act": torch.zeros(1, 4, dim)}
    dit = make_dit(dim=dim, n=n)
    dit.blocks[2] = Recorder(store)
    return conftest.fake_model(dit), dit, store


def _activations(seed, steps=2, dim=8):
    """A fixed, prompt-specific sequence of block outputs (one per sampling step)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 4, dim, generator=g) for _ in range(steps)]


def _drive_wrapper(mp, dit, store, acts):
    wrapper = mp.model_options["model_function_wrapper"]
    for a in acts:
        store["act"] = a
        wrapper(make_apply_model(dit), make_c_in(torch.zeros(1, 8), sig=SIG))


PROMPTS = {"with the idea": _activations(1), "without the idea": _activations(2)}


def test_make_axis_equals_collect_plus_vector(helm):
    nodes = helm.nodes
    # reference path: HelmCollect twice (pos, neg) then HelmVector
    mp, dit, store = _model()
    for tag, text in (("pos", "with the idea"), ("neg", "without the idea")):
        patched = nodes.HelmCollect().run(mp, tag, 2)[0]
        _drive_wrapper(patched, dit, store, PROMPTS[text])
    nodes.HelmVector().run("pos", "neg", "ref_axis")
    ref = nodes._load_axis(nodes._resolve_axis("ref_axis"))

    # new path: one node, both passes
    mp2, dit2, store2 = _model()

    def drive(patched, text):
        _drive_wrapper(patched, dit2, store2, PROMPTS[text])

    model_out, name = nodes.HelmMakeAxis().run(mp2, None, None, "with the idea", "without the idea", "new_axis",
                                               block=2, _drive=drive)
    new = nodes._load_axis(nodes._resolve_axis("new_axis"))
    assert name == "new_axis" and model_out is mp2
    assert torch.equal(new, ref), "Make Axis must be byte-identical to Collect + Vector"
    assert new.dtype == torch.float32 and new.shape == (8,)


def test_axis_is_the_difference_of_the_two_means():
    pos = torch.tensor([3.0, 1.0])
    neg = torch.tensor([1.0, 4.0])
    assert torch.equal(conftest_helm()._axis_from_means(pos, neg), pos - neg)


def conftest_helm():
    return conftest.import_nodes()


def test_saved_axis_is_self_describing(helm):
    nodes = helm.nodes
    mp, dit, store = _model()
    nodes.HelmMakeAxis().run(mp, None, None, "with the idea", "without the idea", "My Fog!", block=2,
                             _drive=lambda p, t: _drive_wrapper(p, dit, store, PROMPTS[t]))
    meta = nodes._read_meta("My_Fog")
    assert meta["kind"] == "axis" and meta["block"] == 2 and meta["dim"] == 8
    assert meta["model"] == "FakeDiT"
    assert meta["from"] == {"pos": "with the idea", "neg": "without the idea"}
    assert meta["safe_max"] > 0 and "estimate" in meta["safe_max_source"]
    header = nodes._load_axis_meta(nodes._resolve_axis("My_Fog"))  # the file alone is complete
    assert header["safe_max"] == meta["safe_max"] and header["block"] == 2


def test_new_axis_shows_up_in_the_dial_and_works(helm):
    nodes = helm.nodes
    mp, dit, store = _model()
    nodes.HelmMakeAxis().run(mp, None, None, "with the idea", "without the idea", "fresh", block=2,
                             _drive=lambda p, t: _drive_wrapper(p, dit, store, PROMPTS[t]))
    assert "Custom · fresh" in nodes._dial_choices()
    dialed = nodes.HelmDial().run(mp, "Custom · fresh", 1.0)[0]
    assert [h["block"] for h in dialed.model_options["helm_hooks"]] == [2]
    # and the Dial can take it straight from Make Axis' output, before any menu refresh
    nodes.HelmDial().run(mp, "(unused)", 0.5, axis_from="fresh")


def test_safe_max_is_a_conservative_fraction_of_the_activation_norm():
    n = conftest_helm()
    pos, neg = torch.full((8,), 10.0), torch.full((8,), 9.0)
    d = float((pos - neg).norm())
    mu = 0.5 * (float(pos.norm()) + float(neg.norm()))
    assert n._estimate_safe_max(pos, neg) == pytest.approx(round(0.005 * mu / d, 4), abs=1e-4)
    # a huge axis relative to the activations gets a tiny ceiling; a tiny one is capped
    assert n._estimate_safe_max(pos, pos - 1000) < n._estimate_safe_max(pos, neg)
    assert n._estimate_safe_max(pos, pos - 1e-6) <= n.SAFE_MAX_CEILING


def test_identical_prompts_are_refused_in_plain_words(helm):
    nodes = helm.nodes
    mp, dit, store = _model()
    same = _activations(5)
    with pytest.raises(ValueError, match="same activations"):
        nodes.HelmMakeAxis().run(mp, None, None, "a", "a", "same", block=2,
                                 _drive=lambda p, t: _drive_wrapper(p, dit, store, same))


def test_default_block_is_the_middle_of_an_uncalibrated_model(helm):
    nodes = helm.nodes
    mp, dit, store = _model(n=6)
    dit.blocks[3] = Recorder(store)
    nodes.HelmMakeAxis().run(mp, None, None, "p", "q", "mid",
                             _drive=lambda p, t: _drive_wrapper(p, dit, store, PROMPTS["with the idea" if t == "p" else "without the idea"]))
    assert nodes._read_meta("mid")["block"] == 3


def test_a_block_that_does_not_exist_is_refused(helm):
    mp, dit, store = _model(n=6)
    with pytest.raises(ValueError, match="does not exist"):
        helm.nodes.HelmMakeAxis().run(mp, None, None, "p", "q", "x", block=99, _drive=lambda p, t: None)


def test_a_sampler_that_never_hits_the_block_is_reported(helm):
    mp, dit, store = _model()
    with pytest.raises(RuntimeError, match="no activations were recorded"):
        helm.nodes.HelmMakeAxis().run(mp, None, None, "p", "q", "x", block=2, _drive=lambda p, t: None)


@pytest.mark.parametrize("name,safe", [("my axis", "my_axis"), ("  fog/dawn  ", "fog_dawn"), ("a-b_c", "a-b_c")])
def test_axis_names_become_safe_file_names(name, safe):
    assert conftest_helm()._safe_axis_name(name) == safe


def test_empty_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        conftest_helm()._safe_axis_name("  ///  ")


def test_make_axis_does_not_leave_hooks_on_the_returned_model(helm):
    mp, dit, store = _model()
    helm.nodes.HelmMakeAxis().run(mp, None, None, "with the idea", "without the idea", "clean", block=2,
                                  _drive=lambda p, t: _drive_wrapper(p, dit, store, PROMPTS[t]))
    assert not mp.model_options.get("helm_hooks"), "the model handed on must be unhooked"


def test_make_axis_writes_only_to_the_user_axis_dir(helm):
    import os
    mp, dit, store = _model()
    helm.nodes.HelmMakeAxis().run(mp, None, None, "with the idea", "without the idea", "where", block=2,
                                  _drive=lambda p, t: _drive_wrapper(p, dit, store, PROMPTS[t]))
    assert os.path.exists(os.path.join(helm.steer, "where.safetensors"))
    assert not os.path.exists(os.path.join(helm.pack, "example_axes", "where.safetensors"))
