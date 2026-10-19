"""Example workflows parse, Helm widget shapes match the frozen order, H3 chain 901 -> 902, and every shipped file satisfies the pack's INPUT_TYPES()."""
import glob
import json
import numbers
import os

import pytest

WIDGET_ORDER = {
    # `model` is a MODEL link input, not a widget value; only widget values are serialized.
    "HelmCollect": [("tag", str), ("block", int), ("cond_only", bool)],
    "HelmVector": [("pos_tag", str), ("neg_tag", str), ("axis", str), ("safe_max", float)],
    "HelmApply": [
        ("axis", str),
        ("block", int),
        ("strength", float),
        ("dial", float),
        ("cond_only", bool),
        ("start_percent", float),
        ("end_percent", float),
    ],
}


def _iter_helm_objs(obj):
    if isinstance(obj, dict):
        t = obj.get("type") or obj.get("class_type")
        if isinstance(t, str) and t.startswith("Helm"):
            yield t, obj
        for v in obj.values():
            yield from _iter_helm_objs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_helm_objs(v)


def _workflow_files(root):
    files = sorted(glob.glob(os.path.join(root, "example_workflows", "*.json")))
    files += sorted(glob.glob(os.path.join(root, "api", "*.json")))
    return files


def test_every_workflow_parses(helm):
    files = _workflow_files(helm.pack)
    assert files, "no workflow files found"
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)


def test_helm_widget_shapes(helm):
    checked = 0
    for path in _workflow_files(helm.pack):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls, node in _iter_helm_objs(data):
            assert cls in WIDGET_ORDER, f"{os.path.basename(path)}: unexpected Helm node {cls}"
            order = WIDGET_ORDER[cls]
            wv = node.get("widgets_values", [])
            assert len(wv) <= len(order), f"{cls} in {os.path.basename(path)} has {len(wv)} widgets > {len(order)}"
            for (name, typ), value in zip(order, wv):
                if typ is float:
                    assert isinstance(value, numbers.Real) and not isinstance(value, bool), (
                        f"{cls}.{name} in {os.path.basename(path)}: {value!r} is not numeric"
                    )
                elif typ is bool:
                    assert isinstance(value, bool), f"{cls}.{name}: {value!r} is not bool"
                else:
                    assert isinstance(value, typ), f"{cls}.{name}: {value!r} is not {typ.__name__}"
            checked += 1
    assert checked > 0, "no Helm nodes found in any workflow"


def test_helm_class_types_exist_in_mappings(helm):
    """Every Helm node in every shipped file names a class registered in NODE_CLASS_MAPPINGS
    (missing keys = exporting a graph against a drifted/renamed node list)."""
    checked = 0
    for path in _workflow_files(helm.pack):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls, node in _iter_helm_objs(data):
            assert cls in helm.nodes.NODE_CLASS_MAPPINGS, (
                f"{os.path.basename(path)}: unknown Helm class {cls} (id {node.get('id', '?')})"
            )
            checked += 1
    assert checked > 0, "no Helm nodes found in any workflow"


def _is_api_format(data):
    # API format: top-level keys are node ids, values carry `class_type`; UI graphs use "type".
    return isinstance(data, dict) and any(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    )


def test_api_helm_nodes_satisfy_required_inputs(helm):
    """Every API-format Helm node supplies each required INPUT_TYPES key; ComfyUI rejects the
    prompt with `required_input_missing` otherwise (`model` is a MODEL link, not a widget value)."""
    checked = 0
    for path in _workflow_files(helm.pack):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not _is_api_format(data):
            continue
        for nid, node in data.items():
            if not (isinstance(node, dict) and str(node.get("class_type", "")).startswith("Helm")):
                continue
            cls = node["class_type"]
            # `model` is a MODEL link, not a widget value; skip it like WIDGET_ORDER does.
            required = {k: v for k, v in helm.nodes.NODE_CLASS_MAPPINGS[cls].INPUT_TYPES()["required"].items()
                        if v[0] != "MODEL"}
            for key in required:
                assert key in node["inputs"], (
                    f"{os.path.basename(path)} node {nid} ({cls}): missing required input '{key}' "
                    f"-> ComfyUI would reject this prompt with required_input_missing"
                )
            checked += 1
    assert checked > 0, "no API-format workflow files found"


def test_h3_t2v_chains_two_helmapply(helm):
    path = os.path.join(helm.pack, "example_workflows", "h3_t2v_helm.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    helm_nodes = [n for n in data["nodes"] if str(n.get("type", "")).startswith("Helm")]
    assert len(helm_nodes) == 2
    ids = {n["id"] for n in helm_nodes}
    assert {901, 902} <= ids

    node902 = next(n for n in data["nodes"] if n["id"] == 902)
    model_input = next(i for i in node902["inputs"] if i["name"] == "model")
    link = next(l for l in data["links"] if l[0] == model_input["link"])
    assert link[1] == 901, "902.model is not fed by 901"
    assert link[3] == 902
