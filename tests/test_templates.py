"""The three starter templates: flat, readable, Helm controls on the canvas, only core + Helm nodes.

Set COMFYUI_DIR to a ComfyUI checkout to also check every node type against that core's source.
"""
import glob
import json
import os
import re

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WF = os.path.join(PACK, "example_workflows")
STARTERS = {
    "helm_1_start_here.json": ("Z-Image", 3840),
    "helm_2_video_h3.json": ("MiniMax H3", 5376),
    "helm_3_make_your_own_axis.json": ("Z-Image", 3840),
}
HELM_TYPES = {"HelmDial", "HelmMakeAxis"}
# Types the starters use from ComfyUI core (checked against core 0.43.2 source). `Note` is a
# frontend-native node, so it has no Python definition.
CORE_TYPES = {
    "BasicGuider", "BasicScheduler", "CLIPLoader", "CLIPTextEncode", "ConditioningZeroOut", "CreateVideo",
    "EmptySD3LatentImage", "ImageStitch", "KSampler", "KSamplerSelect", "LoraLoaderModelOnly",
    "MiniMaxH3ImageToVideo", "ModelSamplingAuraFlow", "PreviewImage", "PrimitiveInt", "RandomNoise",
    "SamplerCustomAdvanced", "SaveImage", "SaveVideo", "UNETLoader", "VAEDecode", "VAEDecodeAudio", "VAELoader",
}
FRONTEND_TYPES = {"Note"}


def _load(name):
    with open(os.path.join(WF, name), encoding="utf-8") as f:
        return json.load(f)


def _all_json():
    return sorted(glob.glob(os.path.join(WF, "**", "*.json"), recursive=True)) + \
        sorted(glob.glob(os.path.join(PACK, "api", "*.json")))


def test_exactly_three_starter_templates_in_the_browser_folder():
    flat = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WF, "*.json")))
    assert flat == sorted(STARTERS), "ComfyUI's template browser lists only example_workflows/*.json"


def test_each_starter_has_a_preview():
    for name in STARTERS:
        assert os.path.exists(os.path.join(WF, name[:-5] + ".png")), name


def test_older_templates_live_in_legacy():
    legacy = glob.glob(os.path.join(WF, "legacy", "*.json"))
    assert legacy, "older templates should stay loadable under legacy/"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_flat_graph_with_helm_on_the_canvas(name):
    data = _load(name)
    assert not data.get("definitions", {}).get("subgraphs"), "no subgraph may hide Helm controls"
    top = {n["type"] for n in data["nodes"]}
    assert "HelmDial" in top, "the Dial must be a top-level node"
    assert not any(re.fullmatch(r"[0-9a-f-]{36}", t) for t in top), "subgraph instance found"
    if name == "helm_3_make_your_own_axis.json":
        assert "HelmMakeAxis" in top
    assert "HelmApply" not in top, "starters use the Dial, not the manual Apply node"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_dial_is_labelled_and_colored(name):
    data = _load(name)
    dial = next(n for n in data["nodes"] if n["type"] == "HelmDial")
    assert dial["title"] == "Helm · Dial"
    assert dial.get("color") and dial.get("bgcolor")
    note = [n for n in data["nodes"] if n["type"] == "Note"]
    assert note, "every starter explains itself with a Note"


def test_template_1_note_is_the_three_steps():
    notes = [n for n in _load("helm_1_start_here.json")["nodes"] if n["type"] == "Note"]
    steps = next(n for n in notes if n.get("title", "").startswith("Helm 1"))
    assert steps["widgets_values"][0].splitlines() == ["1) write your prompt", "2) move Amount", "3) Run"]


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_groups_have_plain_titles(name):
    titles = [g["title"] for g in _load(name)["groups"]]
    assert len(titles) >= 4
    for t in titles:
        assert re.match(r"^\d · [A-Z]", t), t
        for jargon in ("HelmApply", "block", "dim ", "safe_max", "lambda", "diff-in-means", "subgraph"):
            assert jargon not in t, (t, jargon)


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_notes_describe_the_action_not_the_mechanism(name):
    for n in _load(name)["nodes"]:
        if n["type"] != "Note":
            continue
        text = n["widgets_values"][0]
        for jargon in ("HelmApply", "block 15", "dim 3840", "safe_max", "ModelSamplingAuraFlow", "λ", "lambda"):
            assert jargon not in text, (name, jargon)


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_before_after_uses_only_core_stitch(name):
    data = _load(name)
    types = [n["type"] for n in data["nodes"]]
    assert types.count("ImageStitch") == 1
    stitch = next(n for n in data["nodes"] if n["type"] == "ImageStitch")
    assert all(i["link"] is not None for i in stitch["inputs"] if i["name"] in ("image1", "image2"))


@pytest.mark.parametrize("name", ["helm_1_start_here.json", "helm_3_make_your_own_axis.json"])
def test_before_and_after_share_one_seed_and_differ_only_by_the_dial(name):
    data = _load(name)
    links = {l[0]: l for l in data["links"]}
    samplers = [n for n in data["nodes"] if n["type"] == "KSampler"]
    assert len(samplers) == 2
    seeds = {links[next(i["link"] for i in s["inputs"] if i["name"] == "seed")][1] for s in samplers}
    assert len(seeds) == 1, "Before and After must read the same seed node"
    src = {s["title"]: links[next(i["link"] for i in s["inputs"] if i["name"] == "model")][1] for s in samplers}
    dial = next(n["id"] for n in data["nodes"] if n["type"] == "HelmDial")
    assert src["After (your Amount)"] == dial
    assert src["Before (Dial off = Amount 0)"] != dial


def test_template_3_chains_make_axis_into_the_dial():
    data = _load("helm_3_make_your_own_axis.json")
    links = {l[0]: l for l in data["links"]}
    ma = next(n for n in data["nodes"] if n["type"] == "HelmMakeAxis")
    dial = next(n for n in data["nodes"] if n["type"] == "HelmDial")
    feeders = {i["name"]: links[i["link"]][1] for i in dial["inputs"] if i["link"] is not None}
    assert feeders["model"] == ma["id"] and feeders["axis_from"] == ma["id"]


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_every_link_is_consistent(name):
    data = _load(name)
    nodes = {n["id"]: n for n in data["nodes"]}
    assert len(nodes) == len(data["nodes"]), "duplicate node ids"
    seen = set()
    for lid, a, aslot, b, bslot, typ in data["links"]:
        assert lid not in seen
        seen.add(lid)
        assert lid in nodes[a]["outputs"][aslot]["links"], (lid, "origin does not list the link")
        assert nodes[b]["inputs"][bslot]["link"] == lid, (lid, "target does not point back")
        assert nodes[a]["outputs"][aslot]["type"] == nodes[b]["inputs"][bslot]["type"] == typ
    for n in nodes.values():
        for i in n["inputs"]:
            assert i["link"] is None or i["link"] in seen
        for o in n["outputs"]:
            assert all(l in seen for l in o["links"])
    assert data["last_link_id"] >= max(seen) and data["last_node_id"] >= max(nodes)


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_required_inputs_are_wired(name):
    """Every non-optional link-type input of every node is connected (otherwise Run fails at once)."""
    data = _load(name)
    for n in data["nodes"]:
        for i in n["inputs"]:
            if i.get("widget") or i.get("shape") == 7:
                continue
            assert i["link"] is not None, f"{name}: {n['type']} '{n.get('title')}' input '{i['name']}' is unconnected"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_node_types_exist_in_core_or_helm(name, helm):
    for n in _load(name)["nodes"]:
        t = n["type"]
        if t in HELM_TYPES:
            assert t in helm.nodes.NODE_CLASS_MAPPINGS
        else:
            assert t in CORE_TYPES or t in FRONTEND_TYPES, f"{name}: unknown node type {t}"


def test_core_types_exist_in_a_real_comfyui_checkout_if_available():
    root = os.environ.get("COMFYUI_DIR")
    if not root or not os.path.isdir(root):
        pytest.skip("set COMFYUI_DIR to a ComfyUI checkout to verify node types against its source")
    src = ""
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if "custom_nodes" in path or os.sep + "tests" in path:
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            src += f.read()
    known = set(re.findall(r'node_id\s*=\s*"(\w+)"', src)) | set(re.findall(r'^\s*"(\w+)":\s*\w+,?\s*$', src, re.M))
    missing = CORE_TYPES - known
    assert not missing, f"not found in this ComfyUI: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_dial_axis_choice_is_a_menu_entry_that_fits_the_template_model(name, helm):
    family, dim = STARTERS[name]
    dial = next(n for n in _load(name)["nodes"] if n["type"] == "HelmDial")
    choice, amount = dial["widgets_values"]
    assert 0.0 <= amount <= 1.0
    if name == "helm_3_make_your_own_axis.json":
        assert choice.startswith("Custom · ")  # made by the Make Axis node next to it
        return
    assert choice in helm.nodes._dial_choices()
    entry = helm.nodes._dial_resolve(choice)
    assert entry["family"] == family
    vec = helm.nodes._load_axis(helm.nodes._resolve_axis(entry["file"]))
    assert vec.shape[0] == dim


def test_template_3_make_axis_default_name_matches_the_dial():
    data = _load("helm_3_make_your_own_axis.json")
    ma = next(n for n in data["nodes"] if n["type"] == "HelmMakeAxis")
    dial = next(n for n in data["nodes"] if n["type"] == "HelmDial")
    assert dial["widgets_values"][0] == "Custom · " + ma["widgets_values"][2]


def test_no_bypass_lora_anywhere_in_the_package():
    needle = re.compile("fe" + "dor|un" + "clocked", re.I)  # split so this file does not name it
    for path in _all_json():
        with open(path, encoding="utf-8") as f:
            assert not needle.search(f.read()), path
    for root, _dirs, files in os.walk(PACK):
        if ".git" in root.split(os.sep) or os.sep + "tests" in root:
            continue
        for fn in files:
            if fn.endswith((".py", ".md", ".js", ".json", ".toml", ".txt")):
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    assert not needle.search(f.read()), os.path.join(root, fn)


def test_legacy_notes_are_plain_language():
    for path in glob.glob(os.path.join(WF, "legacy", "*.json")):
        for n in json.load(open(path, encoding="utf-8"))["nodes"]:
            if n["type"] == "Note":
                text = n["widgets_values"][0]
                for jargon in ("inserted between", "dim 5376", "block 20, strength", "safe_max"):
                    assert jargon not in text, (os.path.basename(path), jargon)
