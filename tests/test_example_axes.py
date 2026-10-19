"""Guards for the axes that SHIP in example_axes/ (a judge clones the repo and opens a template).

The axes are artifacts, not fixtures: if one is a unit-test stub or loses its header metadata, the
flagship H3 workflow fails on `HelmApply mood -> HelmApply density`. These tests read the files
straight from disk (no torch needed for the header) and cross-check the dimension against
calibration.json.
"""
import json
import os
import struct

import nodes


def _pack_dir():
    return os.path.dirname(os.path.realpath(nodes.__file__))


def _axes_dir():
    return os.path.join(_pack_dir(), "example_axes")


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _shipped_axes():
    return sorted(f for f in os.listdir(_axes_dir()) if f.endswith(".safetensors"))


def _calibration_dims():
    with open(os.path.join(_pack_dir(), "calibration.json"), encoding="utf-8") as f:
        cal = json.load(f)
    return {int(m["dim"]) for m in cal.get("models", []) if m.get("dim")}


def test_every_shipped_axis_is_a_well_formed_activation_axis():
    files = _shipped_axes()
    assert files, "no shipped axes found in example_axes/"
    dims = _calibration_dims()
    for fn in files:
        hdr = _header(os.path.join(_axes_dir(), fn))
        axis = hdr.get("axis")
        assert axis, f"{fn}: no 'axis' tensor"
        assert axis.get("dtype") == "F32", f"{fn}: dtype {axis.get('dtype')}"
        shape = axis.get("shape") or []
        assert len(shape) == 1, f"{fn}: expected a 1-D axis, got shape {shape}"
        dim = int(shape[0])
        meta = hdr.get("__metadata__") or {}
        assert meta.get("kind") == "axis", f"{fn}: missing kind=axis metadata"
        assert int(meta.get("dim", -1)) == dim, (
            f"{fn}: metadata dim {meta.get('dim')} != tensor dim {dim}"
        )
        assert meta.get("block") is not None, f"{fn}: no block in metadata"
        assert meta.get("model"), f"{fn}: no model in metadata"
        assert dim in dims, f"{fn}: dim {dim} is not a calibrated model dim {sorted(dims)}"


def test_shipped_axes_are_not_unit_test_stubs():
    """torch.ones(6) is a test stub, never a shipped artifact (see 3003fc8)."""
    for fn in _shipped_axes():
        dim = int(_header(os.path.join(_axes_dir(), fn))["axis"]["shape"][0])
        assert dim >= 256, f"{fn}: looks like a test stub (dim {dim})"
