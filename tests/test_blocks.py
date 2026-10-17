"""`_find_blocks` attr probing + error reporting (spec 1.3)."""
import types

import pytest
import torch


def _wrap(dm):
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm))


def _ml(n=6):
    return torch.nn.ModuleList([torch.nn.Identity() for _ in range(n)])


@pytest.mark.parametrize("attr", ["blocks", "layers", "transformer_blocks", "double_blocks", "single_blocks"])
def test_finds_each_known_attr(attr):
    import conftest

    nodes = conftest.import_nodes()
    dm = types.SimpleNamespace(**{attr: _ml()})
    ml, name = nodes._find_blocks(_wrap(dm))
    assert name == attr
    assert ml is getattr(dm, attr)


def test_prefers_blocks_over_layers_by_default():
    import conftest

    nodes = conftest.import_nodes()
    blocks, layers = _ml(), _ml()
    dm = types.SimpleNamespace(blocks=blocks, layers=layers)
    ml, name = nodes._find_blocks(_wrap(dm))
    assert name == "blocks"
    assert ml is blocks


def test_which_overrides_probe_order():
    import conftest

    nodes = conftest.import_nodes()
    blocks, layers = _ml(), _ml()
    dm = types.SimpleNamespace(blocks=blocks, layers=layers)
    ml, name = nodes._find_blocks(_wrap(dm), which="layers")
    assert name == "layers"
    assert ml is layers


def test_error_lists_present_attrs():
    import conftest

    nodes = conftest.import_nodes()
    dm = types.SimpleNamespace(layers=_ml(2))
    with pytest.raises(RuntimeError) as ei:
        nodes._find_blocks(_wrap(dm))
    assert "layers" in str(ei.value)


def test_error_when_no_known_attr():
    import conftest

    nodes = conftest.import_nodes()
    dm = types.SimpleNamespace(something_else=_ml())
    with pytest.raises(RuntimeError) as ei:
        nodes._find_blocks(_wrap(dm))
    assert "no transformer block list" in str(ei.value).lower() or "looked for" in str(ei.value).lower()
