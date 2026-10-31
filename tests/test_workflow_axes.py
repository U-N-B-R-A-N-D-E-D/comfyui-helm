"""Workflow axes must match the model the workflow runs.

A shipped template that points at a calibrated axis from another architecture is a runtime error
waiting for the first user. These tests keep every `HelmApply` in the example and API workflows on
an axis that is either shipped with the right dimension or explicitly a build-it-first placeholder.
"""
import glob
import json
import os
import struct

PACK = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AXES = os.path.join(PACK, "example_axes")

MODEL_DIM = {
    "example_workflows/legacy/h3_t2v_helm.json": 5376,
    "example_workflows/legacy/h3_i2v_helm.json": 5376,
    "api/h3_t2v_helm_api.json": 5376,
}

PLACEHOLDER_AXIS = {
    "example_workflows/legacy/krea2_helm.json": "krea2_axis",
    "example_workflows/legacy/krea2_turbo_helm.json": "krea2_axis",
    "api/krea2_helm_api.json": "krea2_axis",
    "example_workflows/legacy/ltx_t2v_helm.json": "ltx_axis",
}


def _axis_dim(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return int(header["axis"]["shape"][0])


def _shipped_dims():
    dims = {}
    for path in glob.glob(os.path.join(AXES, "*.safetensors")):
        dims[os.path.splitext(os.path.basename(path))[0]] = _axis_dim(path)
    return dims


def _workflow_files():
    files = sorted(glob.glob(os.path.join(PACK, "example_workflows", "*.json")))
    files += sorted(glob.glob(os.path.join(PACK, "example_workflows", "legacy", "*.json")))
    files += sorted(glob.glob(os.path.join(PACK, "api", "*.json")))
    return files


def _iter_helmapply(obj):
    if isinstance(obj, dict):
        kind = obj.get("type") or obj.get("class_type")
        if kind == "HelmApply":
            yield obj
        for value in obj.values():
            yield from _iter_helmapply(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_helmapply(value)


def _axis(node):
    if "class_type" in node:
        return node.get("inputs", {}).get("axis")
    widgets = node.get("widgets_values") or []
    return widgets[0] if widgets else None


def test_every_helmapply_axis_matches_its_workflow_model():
    dims = _shipped_dims()
    checked = 0
    for path in _workflow_files():
        rel = os.path.relpath(path, PACK)
        applies = list(_iter_helmapply(json.load(open(path, encoding="utf-8"))))
        if not applies:
            continue
        assert rel in MODEL_DIM or rel in PLACEHOLDER_AXIS, (
            f"{rel}: HelmApply has no expected axis mapping"
        )
        for node in applies:
            axis = _axis(node)
            assert isinstance(axis, str) and axis, f"{rel}: HelmApply has no axis"
            if rel in PLACEHOLDER_AXIS:
                assert axis == PLACEHOLDER_AXIS[rel], (
                    f"{rel}: expected build-it-first axis {PLACEHOLDER_AXIS[rel]!r}, got {axis!r}"
                )
                assert axis not in dims, f"{rel}: placeholder {axis!r} must not ship an axis"
            else:
                assert axis in dims, f"{rel}: axis {axis!r} is not shipped in example_axes/"
                assert dims[axis] == MODEL_DIM[rel], (
                    f"{rel}: axis {axis!r} has dim {dims[axis]}, expected {MODEL_DIM[rel]}"
                )
            checked += 1
    assert checked > 0, "no HelmApply nodes found in the shipped workflows"


def test_every_shipped_axis_has_a_calibrated_workflow_dimension():
    dims = _shipped_dims()
    # the flat starter templates use HelmDial (tests/test_templates.py checks those axes): Z-Image + H3
    expected = set(MODEL_DIM.values()) | {3840, 5376}
    for axis, dim in dims.items():
        assert dim in expected, f"{axis}: dim {dim} is not used by a calibrated workflow"
