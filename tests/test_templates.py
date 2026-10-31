"""The six starter templates: flat, readable, Helm controls on the canvas, only core + Helm nodes.

Set COMFYUI_DIR to a ComfyUI checkout to also check every node type against that core's source.
"""
import glob
import json
import os
import re

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WF = os.path.join(PACK, "example_workflows")
# ComfyUI's template browser shows custom-node templates by FILE NAME (there is no index for them), and looks
# for "<name>.jpg" next to the workflow as the card thumbnail. So the names below are the titles people read.
T1 = "Helm 1. Start here (Z-Image).json"
T2 = "Helm 2. Video (MiniMax H3).json"
T3 = "Helm 3. Make your own axis.json"
T4 = "Helm 4. Chain ideas (Axis \u2192 Apply).json"
T5 = "Helm 5. Shape and mix (Axis \u00d7 3 \u2192 Mix \u2192 Shape \u2192 Apply).json"
T6 = "Helm 6. Build a multipurpose axis from prompt pairs.json"
DIAL_TEMPLATES = (T1, T2, T3)      # the one-node Dial shortcut
CHAIN_TEMPLATES = (T4, T5, T6)     # the 0.5 chain: Axis, Shape, Mix, Apply
STARTERS = {
    T1: ("Z-Image", 3840),
    T2: ("MiniMax H3", 5376),
    T3: ("Z-Image", 3840),
    T4: ("Z-Image", 3840),
    T5: ("Z-Image", 3840),
    T6: ("Z-Image", 3840),
}
HELM_TYPES = {"HelmDial", "HelmMakeAxis", "HelmAxis", "HelmShape", "HelmMix", "HelmApplyNode"}
# Role colors (title, body): the same table web/helm_axes.js gives a node when you add it from search.
ROLE_COLORS = {
    "HelmAxis": ("#3a2352", "#5b3a80"),        # purple: WHAT
    "HelmShape": ("#6b3300", "#a85200"),       # orange: HOW MUCH
    "HelmDial": ("#6b3300", "#a85200"),        # orange: the one-node shortcut
    "HelmMakeAxis": ("#6b3300", "#a85200"),
    "HelmMix": ("#1c3a5e", "#2f5f94"),         # blue: COMBINE
    "HelmApplyNode": ("#333", "#555"),         # grey: WHERE
}
GROUP_COLORS = {"#3f8e5e", "#b06634", "#8A8A8A", "#3f789e", "#7e57a3"}
# Types the starters use from ComfyUI core (checked against core 0.43.2 source). `MarkdownNote` is a
# frontend-native node, so it has no Python definition.
CORE_TYPES = {
    "BasicGuider", "BasicScheduler", "CLIPLoader", "CLIPTextEncode", "ConditioningZeroOut", "CreateVideo",
    "EmptySD3LatentImage", "ImageStitch", "KSampler", "KSamplerSelect", "LoraLoaderModelOnly",
    "MiniMaxH3ImageToVideo", "ModelSamplingAuraFlow", "PreviewImage", "PrimitiveInt", "RandomNoise",
    "SamplerCustomAdvanced", "SaveImage", "SaveVideo", "UNETLoader", "VAEDecode", "VAEDecodeAudio", "VAELoader",
}
FRONTEND_TYPES = {"MarkdownNote"}


def _load(name):
    with open(os.path.join(WF, name), encoding="utf-8") as f:
        return json.load(f)


def _all_json():
    return sorted(glob.glob(os.path.join(WF, "**", "*.json"), recursive=True)) + \
        sorted(glob.glob(os.path.join(PACK, "api", "*.json")))


def test_exactly_six_starter_templates_in_the_browser_folder():
    flat = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WF, "*.json")))
    assert flat == sorted(STARTERS), "ComfyUI's template browser lists only example_workflows/*.json"


def test_each_starter_has_a_square_jpg_thumbnail_with_the_same_name():
    from PIL import Image
    for name in STARTERS:
        path = os.path.join(WF, name[:-5] + ".jpg")
        assert os.path.exists(path), name
        with Image.open(path) as im:
            assert im.size[0] == im.size[1] >= 400, (name, im.size)
            assert not im.getexif(), "thumbnails carry no metadata"
        assert os.path.getsize(path) < 300_000, name


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_file_name_is_a_readable_title(name):
    title = name[:-5]
    assert re.match(r"^Helm [1-6]\. [A-Z]", title), title
    assert "_" not in title
    assert set(c for c in title if not c.isascii()) <= {"\u2192", "\u00d7"}, "only the arrow and the times sign"


def test_older_templates_live_in_legacy():
    legacy = glob.glob(os.path.join(WF, "legacy", "*.json"))
    assert legacy, "older templates should stay loadable under legacy/"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_flat_graph_with_helm_on_the_canvas(name):
    data = _load(name)
    assert not data.get("definitions", {}).get("subgraphs"), "no subgraph may hide Helm controls"
    top = {n["type"] for n in data["nodes"]}
    if name in DIAL_TEMPLATES:
        assert "HelmDial" in top, "the Dial must be a top-level node"
    else:
        assert {"HelmAxis", "HelmApplyNode"} <= top, "the chain templates show Axis and Apply on the canvas"
        assert "HelmDial" not in top
    assert not any(re.fullmatch(r"[0-9a-f-]{36}", t) for t in top), "subgraph instance found"
    if name == T3:
        assert "HelmMakeAxis" in top
    assert "HelmApply" not in top, "starters use the Dial, not the manual Apply node"


@pytest.mark.parametrize("name", DIAL_TEMPLATES)
def test_dial_is_labelled_and_colored(name):
    data = _load(name)
    dial = next(n for n in data["nodes"] if n["type"] == "HelmDial")
    assert dial["title"] == "Helm · Dial"
    assert dial.get("color") and dial.get("bgcolor")
    note = [n for n in data["nodes"] if n["type"] == "MarkdownNote"]
    assert note, "every starter explains itself with a MarkdownNote"
    assert not [n for n in data["nodes"] if n["type"] == "Note"], "one note type only: MarkdownNote, as in the official templates"
    assert dial["size"][0] >= 500, "the Dial is the biggest thing a first-time user should see"


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_the_three_steps_are_the_first_thing_on_the_canvas(name):
    data = _load(name)
    notes = [n for n in data["nodes"] if n["type"] == "MarkdownNote"]
    steps = next(n for n in notes if n.get("title") == "Note: Start here")
    lines = [l for l in steps["widgets_values"][0].splitlines() if l.strip()]
    assert lines[0] == "# Start here"
    assert [l[:6] for l in lines[1:]] == ["## 1. ", "## 2. ", "## 3. "]
    assert steps["pos"] == [0, 0], "top-left, where the eye starts"
    assert all(n["pos"][0] >= 0 and n["pos"][1] >= 0 for n in data["nodes"])
    assert min(data["nodes"], key=lambda n: (n["pos"][1], n["pos"][0])) is steps


def test_template_1_steps_are_write_move_run():
    notes = [n for n in _load(T1)["nodes"] if n["type"] == "MarkdownNote"]
    text = next(n for n in notes if n["title"] == "Note: Start here")["widgets_values"][0]
    assert "Write your prompt" in text and "Move **Amount**" in text and "**Run**" in text
    assert text.index("Write") < text.index("Amount") < text.index("Run")


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_notes_are_short_and_scannable_and_use_the_official_look(name):
    for n in _load(name)["nodes"]:
        if n["type"] != "MarkdownNote":
            continue
        text = n["widgets_values"][0]
        assert n["title"].startswith("Note: ")
        assert text.startswith("#"), "every note opens with a heading"
        assert len(text) < 750, (n["title"], len(text))
        assert (n["color"], n["bgcolor"]) == ("#222", "#000"), "same colors as the official MarkdownNote"
        assert n["properties"] == {}


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_node_colors_match_group_colors(name):
    """Green = you edit it, orange = the Helm control, grey = leave alone, blue = the result."""
    data = _load(name)
    colors = {g["title"].split(" · ")[-1].split(" (")[0]: g["color"] for g in data["groups"]}
    assert {"#3f8e5e", "#8A8A8A", "#3f789e"} <= set(g["color"] for g in data["groups"]) <= GROUP_COLORS
    for n in data["nodes"]:
        if n["type"] in HELM_TYPES:
            assert (n["color"], n["bgcolor"]) == ROLE_COLORS[n["type"]], (name, n.get("title"))
            grp = next(g for g in data["groups"] if _inside(n, g))
            if n["type"] == "HelmAxis":
                assert grp["color"] == "#7e57a3", grp["title"]  # Axis lives in the purple group
            elif n["type"] == "HelmMakeAxis":
                assert grp["color"] in ("#b06634", "#3f8e5e"), grp["title"]  # Make Axis sits in the green "describe" group
            else:
                assert grp["color"] in ("#b06634", "#7e57a3"), grp["title"]
        if n.get("title") == "Your prompt (edit me)":
            assert (n["color"], n["bgcolor"]) == ("#232", "#353")
            assert next(g for g in data["groups"] if _inside(n, g))["color"] == "#3f8e5e"
    assert colors


def _footprint(node):
    """What the node occupies on screen: collapsed nodes are one-line pills (the JSON keeps the expanded size)."""
    return (300, 40) if node.get("flags", {}).get("collapsed") else tuple(node["size"])


def _inside(node, group):
    x, y = node["pos"]
    w, h = _footprint(node)
    gx, gy, gw, gh = group["bounding"]
    return gx <= x and gy <= y and x + w <= gx + gw and y + h <= gy + gh


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_no_node_overlaps_another_or_leaves_its_group(name):
    data = _load(name)
    nodes = data["nodes"]
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            ax, ay = a["pos"]; bx, by = b["pos"]
            (aw, ah), (bw, bh) = _footprint(a), _footprint(b)
            overlap = ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            assert not overlap, (a.get("title"), b.get("title"))
    for g in data["groups"]:
        inside = [n for n in nodes if _inside(n, g)]
        assert inside, g["title"]


def test_node_help_exists_for_every_main_node():
    for name in ("HelmDial", "HelmMakeAxis", "HelmAxis", "HelmShape", "HelmMix", "HelmApplyNode"):
        text = open(os.path.join(PACK, "web", "docs", name + ".md"), encoding="utf-8").read()
        assert text.startswith("# Helm · ") and len(text) < 2500


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_groups_have_plain_titles(name):
    titles = [g["title"] for g in _load(name)["groups"]]
    assert len(titles) >= 4
    steps = [t for t in titles if t[0].isdigit()]
    assert [int(t[0]) for t in steps] == list(range(1, len(steps) + 1)), "numbered steps, in order"
    assert [t for t in titles if not t[0].isdigit()] == ["Models (leave as they are)"]
    for t in titles:
        assert re.match(r"^(\d · )?[A-Z]", t), t
        for jargon in ("HelmApply", "block", "dim ", "safe_max", "lambda", "diff-in-means", "subgraph"):
            assert jargon not in t, (t, jargon)


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_notes_describe_the_action_not_the_mechanism(name):
    for n in _load(name)["nodes"]:
        if n["type"] != "MarkdownNote":
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


@pytest.mark.parametrize("name", [T1, T3, T4, T5, T6])
def test_before_and_after_share_one_seed_and_differ_only_by_the_helm_control(name):
    data = _load(name)
    links = {l[0]: l for l in data["links"]}
    samplers = [n for n in data["nodes"] if n["type"] == "KSampler"]
    assert len(samplers) == 2
    seeds = {links[next(i["link"] for i in s["inputs"] if i["name"] == "seed")][1] for s in samplers}
    assert len(seeds) == 1, "Before and After must read the same seed node"
    src = {s["title"]: links[next(i["link"] for i in s["inputs"] if i["name"] == "model")][1] for s in samplers}
    last = "HelmDial" if name in DIAL_TEMPLATES else "HelmApplyNode"
    dial = next(n["id"] for n in data["nodes"] if n["type"] == last)
    assert src["After (your Amount)"] == dial
    assert src["Before (Dial off)"] != dial


def test_template_3_chains_make_axis_into_the_dial():
    data = _load(T3)
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


def _chain(name):
    """node type -> list of (node, {input name: id of the node feeding it}) for a chain template."""
    data = _load(name)
    links = {l[0]: l for l in data["links"]}
    nodes = {n["id"]: n for n in data["nodes"]}
    out = {}
    for n in data["nodes"]:
        feeders = {i["name"]: nodes[links[i["link"]][1]] for i in n["inputs"] if i["link"] is not None}
        out.setdefault(n["type"], []).append((n, feeders))
    return out


def test_template_4_is_the_two_node_path_axis_to_apply():
    chain = _chain(T4)
    assert len(chain["HelmAxis"]) == 1 and len(chain["HelmApplyNode"]) == 1
    assert not {"HelmShape", "HelmMix"} & set(chain)
    apply_, feeders = chain["HelmApplyNode"][0]
    assert feeders["helm"]["type"] == "HelmAxis"


def test_template_5_chains_three_axes_into_mix_then_shape_then_apply():
    chain = _chain(T5)
    assert len(chain["HelmAxis"]) == 3
    mix, mix_in = chain["HelmMix"][0]
    assert {mix_in[k]["type"] for k in ("helm_1", "helm_2", "helm_3")} == {"HelmAxis"}
    assert len({mix_in[k]["id"] for k in ("helm_1", "helm_2", "helm_3")}) == 3
    shape, shape_in = chain["HelmShape"][0]
    assert shape_in["helm"]["id"] == mix["id"]
    apply_, apply_in = chain["HelmApplyNode"][0]
    assert apply_in["helm"]["id"] == shape["id"]
    for n in (mix, shape):
        assert "experimental" in n["title"].lower(), "Shape and Mix are marked experimental on the canvas"
    notes = " ".join(n["widgets_values"][0] for n in _load(T5)["nodes"] if n["type"] == "MarkdownNote")
    assert "Experimental" in notes and "not yet measured" in notes


def test_template_6_builds_the_axis_from_prompt_pairs_and_applies_it():
    chain = _chain(T6)
    ax, ax_in = chain["HelmAxis"][0]
    source, idea, pairs, name = ax["widgets_values"][:4]
    assert source == "prompts"
    lines = [l for l in pairs.splitlines() if "||" in l]
    assert len(lines) >= 3, "a multipurpose axis needs several pairs on different subjects"
    assert {"model", "clip", "latent"} <= set(ax_in)
    apply_, apply_in = chain["HelmApplyNode"][0]
    assert apply_in["helm"]["id"] == ax["id"]


@pytest.mark.parametrize("name", CHAIN_TEMPLATES)
def test_chain_nodes_use_the_documented_defaults(name):
    for node, _ in sum(_chain(name).values(), []):
        if node["type"] == "HelmApplyNode":
            assert node["widgets_values"] == [False, "push"]
        if node["type"] == "HelmShape":
            assert -1 <= node["widgets_values"][0] <= 1
        if node["type"] == "HelmAxis":
            assert node["widgets_values"][8] == "fixed", "seed does not change between runs"


def test_template_colors_match_the_colors_web_helm_axes_js_gives_a_new_node():
    js = open(os.path.join(PACK, "web", "helm_axes.js"), encoding="utf-8").read()
    for cls, role in (("HelmAxis", "axis"), ("HelmShape", "shape"), ("HelmMix", "mix"), ("HelmApplyNode", "apply"),
                      ("HelmDial", "shape"), ("HelmMakeAxis", "shape")):
        m = re.search(role + r': \{ color: "(#[0-9a-f]+)", bgcolor: "(#[0-9a-f]+)"', js)
        assert m and (m.group(1), m.group(2)) == ROLE_COLORS[cls], cls
        assert re.search(cls + r': "' + role + '"', js), cls


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_axis_choice_is_a_menu_entry_that_fits_the_template_model(name, helm):
    family, dim = STARTERS[name]
    if name in CHAIN_TEMPLATES:
        for node, _ in _chain(name)["HelmAxis"]:
            if node["widgets_values"][0] == "prompts":
                continue
            choice = node["widgets_values"][1]
            entry = helm.nodes._dial_resolve(choice)
            assert choice in helm.nodes._dial_choices() and entry["family"] == family
            assert helm.nodes._load_axis(helm.nodes._resolve_axis(entry["file"])).shape[0] == dim
        return
    dial = next(n for n in _load(name)["nodes"] if n["type"] == "HelmDial")
    choice, amount = dial["widgets_values"]
    assert 0.0 <= amount <= 1.0
    if name == T3:
        assert choice.startswith("Custom · ")  # made by the Make Axis node next to it
        return
    assert choice in helm.nodes._dial_choices()
    entry = helm.nodes._dial_resolve(choice)
    assert entry["family"] == family
    vec = helm.nodes._load_axis(helm.nodes._resolve_axis(entry["file"]))
    assert vec.shape[0] == dim


def test_template_3_make_axis_default_name_matches_the_dial():
    data = _load(T3)
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


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_graph_is_compact_so_it_opens_readable(name):
    """ComfyUI always fits a template to the window and, below about 60% zoom, draws nodes as empty boxes.
    A 1600 x 900 window leaves about 1544 x 860 px; fit-to-view keeps a margin of about 25% (measured); the real browser shows about 4% more zoom than this estimate, and nodes are drawn as empty boxes below 0.6."""
    data = _load(name)
    xs = [n["pos"][0] for n in data["nodes"]] + [g["bounding"][0] for g in data["groups"]]
    xe = [n["pos"][0] + _footprint(n)[0] for n in data["nodes"]] + [g["bounding"][0] + g["bounding"][2] for g in data["groups"]]
    ys = [n["pos"][1] for n in data["nodes"]] + [g["bounding"][1] for g in data["groups"]]
    ye = [n["pos"][1] + _footprint(n)[1] for n in data["nodes"]] + [g["bounding"][1] + g["bounding"][3] for g in data["groups"]]
    width, height = max(xe) - min(xs), max(ye) - min(ys)
    assert min(1544 / width, 860 / height) / 1.25 >= 0.58, (width, height)


@pytest.mark.parametrize("name", sorted(STARTERS))
def test_plumbing_is_collapsed_but_what_you_touch_is_not(name):
    data = _load(name)
    for n in data["nodes"]:
        collapsed = bool(n.get("flags", {}).get("collapsed"))
        if n["type"] in HELM_TYPES or n["type"] == "MarkdownNote" or n.get("title") == "Your prompt (edit me)":
            assert not collapsed, n.get("title")
        if n["type"] in ("UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "KSampler",
                         "VAEDecode", "ImageStitch"):
            assert collapsed, n.get("title")
