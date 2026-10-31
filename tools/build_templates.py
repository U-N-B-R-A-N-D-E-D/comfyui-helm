"""Regenerates the six starter templates in example_workflows/ (flat graphs, no subgraphs).

ComfyUI's template browser lists custom-node templates by FILE NAME (there is no index for them) and looks for
a thumbnail next to the workflow with the same name and a .jpg extension, so the file names below are the
titles people read. See docs/UX-NOTES.md.

    python tools/build_templates.py

The templates are plain ComfyUI "UI format" JSON. They are generated so that the three graphs stay
consistent (same node sizes, colors, group titles) and so that every link is checked once, here and again
by tests/test_templates.py. Only core ComfyUI nodes and Helm nodes are used.

Templates 1 to 3 use the one-node Dial (plus Make Axis in 3). Templates 4 to 6 teach the 0.5 chain by level:
Axis -> Apply, then Axis x 3 -> Mix -> Shape -> Apply, then an Axis built from prompt pairs.
"""
import json
import os
import uuid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows")

# name, type, is_widget, optional
S = {
    "UNETLoader": ([("unet_name", "COMBO", 1), ("weight_dtype", "COMBO", 1)], [("MODEL", "MODEL")]),
    "CLIPLoader": ([("clip_name", "COMBO", 1), ("type", "COMBO", 1), ("device", "COMBO", 1, 1)], [("CLIP", "CLIP")]),
    "VAELoader": ([("vae_name", "COMBO", 1)], [("VAE", "VAE")]),
    "CLIPTextEncode": ([("clip", "CLIP", 0), ("text", "STRING", 1)], [("CONDITIONING", "CONDITIONING")]),
    "ConditioningZeroOut": ([("conditioning", "CONDITIONING", 0)], [("CONDITIONING", "CONDITIONING")]),
    "EmptySD3LatentImage": ([("width", "INT", 1), ("height", "INT", 1), ("batch_size", "INT", 1)], [("LATENT", "LATENT")]),
    "ModelSamplingAuraFlow": ([("model", "MODEL", 0), ("shift", "FLOAT", 1)], [("MODEL", "MODEL")]),
    "KSampler": ([("model", "MODEL", 0), ("positive", "CONDITIONING", 0), ("negative", "CONDITIONING", 0),
                  ("latent_image", "LATENT", 0), ("seed", "INT", 1), ("steps", "INT", 1), ("cfg", "FLOAT", 1),
                  ("sampler_name", "COMBO", 1), ("scheduler", "COMBO", 1), ("denoise", "FLOAT", 1)],
                 [("LATENT", "LATENT")]),
    "VAEDecode": ([("samples", "LATENT", 0), ("vae", "VAE", 0)], [("IMAGE", "IMAGE")]),
    "ImageStitch": ([("image1", "IMAGE", 0), ("direction", "COMBO", 1), ("match_image_size", "BOOLEAN", 1),
                     ("spacing_width", "INT", 1), ("spacing_color", "COMBO", 1), ("image2", "IMAGE", 0, 1)],
                    [("IMAGE", "IMAGE")]),
    "PreviewImage": ([("images", "IMAGE", 0)], []),
    "SaveImage": ([("images", "IMAGE", 0), ("filename_prefix", "STRING", 1)], []),
    "PrimitiveInt": ([("value", "INT", 1)], [("INT", "INT")]),
    "MarkdownNote": ([], []),
    "LoraLoaderModelOnly": ([("model", "MODEL", 0), ("lora_name", "COMBO", 1), ("strength_model", "FLOAT", 1)],
                            [("MODEL", "MODEL")]),
    "MiniMaxH3ImageToVideo": ([("clip", "CLIP", 0), ("vae", "VAE", 0), ("first_frame", "IMAGE", 0, 1),
                               ("last_frame", "IMAGE", 0, 1), ("prompt", "STRING", 1), ("width", "INT", 1),
                               ("height", "INT", 1), ("length", "INT", 1)],
                              [("positive", "CONDITIONING"), ("LATENT", "LATENT")]),
    "KSamplerSelect": ([("sampler_name", "COMBO", 1)], [("SAMPLER", "SAMPLER")]),
    "BasicScheduler": ([("model", "MODEL", 0), ("scheduler", "COMBO", 1), ("steps", "INT", 1), ("denoise", "FLOAT", 1)],
                       [("SIGMAS", "SIGMAS")]),
    "RandomNoise": ([("noise_seed", "INT", 1)], [("NOISE", "NOISE")]),
    "BasicGuider": ([("model", "MODEL", 0), ("conditioning", "CONDITIONING", 0)], [("GUIDER", "GUIDER")]),
    "SamplerCustomAdvanced": ([("noise", "NOISE", 0), ("guider", "GUIDER", 0), ("sampler", "SAMPLER", 0),
                               ("sigmas", "SIGMAS", 0), ("latent_image", "LATENT", 0)],
                              [("output", "LATENT"), ("denoised_output", "LATENT")]),
    "VAEDecodeAudio": ([("samples", "LATENT", 0), ("vae", "VAE", 0)], [("AUDIO", "AUDIO")]),
    "CreateVideo": ([("images", "IMAGE", 0), ("fps", "FLOAT", 1), ("audio", "AUDIO", 0, 1), ("bit_depth", "COMBO", 1, 1),
                     ("color_space", "COMBO", 1, 1), ("codec", "COMBO", 1, 1)], [("VIDEO", "VIDEO")]),
    "SaveVideo": ([("video", "VIDEO", 0), ("filename_prefix", "STRING", 1), ("format", "COMFY_DYNAMICCOMBO_V3", 1),
                   ("codec", "COMFY_DYNAMICCOMBO_V3", 1)], []),
    "HelmDial": ([("model", "MODEL", 0), ("axis", "COMBO", 1), ("amount", "FLOAT", 1), ("axis_from", "STRING", 0, 1)],
                 [("model", "MODEL")]),
    "HelmAxis": ([("helm", "HELM", 0, 1), ("model", "MODEL", 0, 1), ("clip", "CLIP", 0, 1), ("latent", "LATENT", 0, 1),
                   ("source", "COMBO", 1), ("idea", "COMBO", 1), ("pairs", "STRING", 1, 1), ("name", "STRING", 1, 1),
                   ("debias", "BOOLEAN", 1, 1), ("debias_strength", "FLOAT", 1, 1), ("steps", "INT", 1, 1),
                   ("seed", "INT", 1, 1), ("cfg", "FLOAT", 1, 1), ("block", "INT", 1, 1)],
                  [("helm", "HELM")]),
    "HelmShape": ([("helm", "HELM", 0), ("amount", "FLOAT", 1), ("stage", "COMBO", 1, 1), ("split", "FLOAT", 1, 1),
                   ("curve", "COMBO", 1, 1), ("end_amount", "FLOAT", 1, 1), ("easing", "COMBO", 1, 1),
                   ("keyframes", "STRING", 1, 1)],
                  [("helm", "HELM")]),
    "HelmMix": ([("helm_1", "HELM", 0), ("helm_2", "HELM", 0, 1), ("helm_3", "HELM", 0, 1), ("helm_4", "HELM", 0, 1),
                 ("orthogonalize", "BOOLEAN", 1), ("weight_1", "FLOAT", 1, 1), ("weight_2", "FLOAT", 1, 1),
                 ("weight_3", "FLOAT", 1, 1), ("weight_4", "FLOAT", 1, 1), ("axis_weights", "STRING", 1, 1)],
                [("helm", "HELM"), ("report", "STRING")]),
    "HelmApplyNode": ([("model", "MODEL", 0), ("helm", "HELM", 0), ("cond_only", "BOOLEAN", 1, 1), ("mode", "COMBO", 1, 1)],
                      [("model", "MODEL")]),
    "HelmMakeAxis": ([("model", "MODEL", 0), ("clip", "CLIP", 0), ("latent", "LATENT", 0), ("with_idea", "STRING", 1),
                      ("without_idea", "STRING", 1), ("name", "STRING", 1), ("steps", "INT", 1),
                      ("seed", "INT", 1, 1), ("cfg", "FLOAT", 1, 1), ("block", "INT", 1, 1)],
                     [("model", "MODEL"), ("axis", "STRING")]),
}

HELM_NODES = ("HelmDial", "HelmMakeAxis", "HelmAxis", "HelmShape", "HelmMix", "HelmApplyNode")

# colors (title, body). The Helm role colors are the same ones web/helm_axes.js gives a node when you add it from search.
GREEN = ("#232", "#353")      # things you edit
PURPLE = ("#3a2352", "#5b3a80")   # Helm Axis: WHAT
ORANGE = ("#6b3300", "#a85200")   # Helm Shape and the one-node Dial: HOW MUCH
BLUE = ("#1c3a5e", "#2f5f94")     # Helm Mix: COMBINE
GREY = ("#333", "#555")           # Helm Apply: WHERE
NOTE = ("#222", "#000")       # same as the official ComfyUI templates' MarkdownNote
GROUP_BLUE = "#3f789e"
GROUP_ORANGE = "#b06634"
GROUP_GREEN = "#3f8e5e"
GROUP_GREY = "#8A8A8A"
GROUP_PURPLE = "#7e57a3"

Z_MODELS = {
    "UNETLoader": ("z_image_turbo_bf16.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    "CLIPLoader": ("qwen_3_4b.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    "VAELoader": ("ae.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", "vae"),
}


class Graph:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []
        self.nid = self.lid = self.gid = 0
        self.by_id = {}

    def add(self, typ, pos, size, widgets=None, title=None, colors=None, models=None, text=None, collapsed=False):
        self.nid += 1
        ins, outs = S[typ]
        node = {
            "id": self.nid, "type": typ, "pos": list(pos), "size": list(size), "flags": {}, "order": self.nid,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outs],
            "properties": {} if typ == "MarkdownNote" else {"Node name for S&R": typ},
            "widgets_values": widgets if widgets is not None else [],
        }
        for spec in ins:
            name, t, is_w = spec[0], spec[1], spec[2]
            entry = {"name": name, "type": t, "link": None}
            if is_w:
                entry["widget"] = {"name": name}
            if len(spec) > 3 and spec[3]:
                entry["shape"] = 7
            node["inputs"].append(entry)
        if typ in HELM_NODES:
            node["properties"]["cnr_id"] = "comfyui-helm"
        if models:
            node["properties"]["models"] = [{"name": n, "url": u, "directory": d} for n, u, d in models]
        if text is not None:
            node["properties"] = {}  # as in the official templates
            node["widgets_values"] = [text]
        if collapsed:
            node["flags"] = {"collapsed": True}
        if title:
            node["title"] = title
        if colors:
            node["color"], node["bgcolor"] = colors
        self.nodes.append(node)
        self.by_id[node["id"]] = node
        return node["id"]

    def link(self, a, out_name, b, in_name):
        src, dst = self.by_id[a], self.by_id[b]
        oi = next(i for i, o in enumerate(src["outputs"]) if o["name"].lower() == out_name.lower())
        ii = next(i for i, x in enumerate(dst["inputs"]) if x["name"] == in_name)
        assert dst["inputs"][ii]["link"] is None, (b, in_name)
        assert src["outputs"][oi]["type"] == dst["inputs"][ii]["type"], (a, out_name, b, in_name)
        self.lid += 1
        dst["inputs"][ii]["link"] = self.lid
        src["outputs"][oi]["links"].append(self.lid)
        self.links.append([self.lid, a, oi, b, ii, src["outputs"][oi]["type"]])

    def group(self, title, x, y, w, h, color=GROUP_BLUE):
        """Big titles: at the zoom a template opens at, 24 px titles are too small to read."""
        self.gid += 1
        self.groups.append({"id": self.gid, "title": title, "bounding": [x, y, w, h], "color": color,
                            "font_size": 40, "flags": {}})

    def dump(self, path, scale=0.7, offset=(100, 130)):  # ds is ignored for templates (the frontend fits them)
        doc = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(path))), "revision": 0,
               "last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes, "links": self.links,
               "groups": self.groups, "config": {}, "extra": {"ds": {"scale": scale, "offset": list(offset)}},
               "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")


PILL = (300, 40)   # footprint of a collapsed node (the JSON keeps the expanded size)


def z_models(g, x, y):
    """The loaders, collapsed to one-line pills: they are 'leave as they are', so they take no attention."""
    u = g.add("UNETLoader", (x, y), (320, 100), [Z_MODELS["UNETLoader"][0], "default"], "Z-Image model",
              models=[Z_MODELS["UNETLoader"]], collapsed=True)
    c = g.add("CLIPLoader", (x, y + 50), (320, 130), [Z_MODELS["CLIPLoader"][0], "lumina2", "default"],
              "Text encoder", models=[Z_MODELS["CLIPLoader"]], collapsed=True)
    v = g.add("VAELoader", (x, y + 100), (320, 90), [Z_MODELS["VAELoader"][0]], "Image decoder (VAE)",
              models=[Z_MODELS["VAELoader"]], collapsed=True)
    s = g.add("ModelSamplingAuraFlow", (x, y + 150), (300, 100), [3], "Sampling shift", collapsed=True)
    g.link(u, "MODEL", s, "model")
    return u, c, v, s


def before_after_z(g, x, y, model_before, model_after, pos, neg, latent, vae, seed, pw=660):
    """Two KSamplers with the same seed. Before = model straight from the loader chain (Dial off).

    (x, y) is the group origin. The picture comes first; the plumbing is collapsed into pills under it."""
    pv = g.add("PreviewImage", (x + 20, y + 80), (pw, 420), None, "Left: Before    Right: After")
    row = y + 540
    kb = g.add("KSampler", (x + 20, row), (300, 340), [0, "fixed", 8, 1, "res_multistep", "simple", 1],
               "Before (Dial off)", collapsed=True)
    db = g.add("VAEDecode", (x + 340, row), (180, 60), None, "Decode Before", collapsed=True)
    ka = g.add("KSampler", (x + 20, row + 50), (300, 340), [0, "fixed", 8, 1, "res_multistep", "simple", 1],
               "After (your Amount)", collapsed=True)
    da = g.add("VAEDecode", (x + 340, row + 50), (180, 60), None, "Decode After", collapsed=True)
    st = g.add("ImageStitch", (x + 20, row + 100), (300, 190), ["right", True, 0, "white"], "Before | After", collapsed=True)
    sv = g.add("SaveImage", (x + 340, row + 100), (270, 200), ["helm/after"], "Save After image", collapsed=True)
    for k, m in ((kb, model_before), (ka, model_after)):
        g.link(m, "MODEL", k, "model")
        g.link(pos, "CONDITIONING", k, "positive")
        g.link(neg, "CONDITIONING", k, "negative")
        g.link(latent, "LATENT", k, "latent_image")
        g.link(seed, "INT", k, "seed")
    g.link(kb, "LATENT", db, "samples"); g.link(vae, "VAE", db, "vae")
    g.link(ka, "LATENT", da, "samples"); g.link(vae, "VAE", da, "vae")
    g.link(db, "IMAGE", st, "image1"); g.link(da, "IMAGE", st, "image2")
    g.link(st, "IMAGE", pv, "images")
    g.link(da, "IMAGE", sv, "images")
    return kb, ka


T1 = "Helm 1. Start here (Z-Image).json"
T2 = "Helm 2. Video (MiniMax H3).json"
T3 = "Helm 3. Make your own axis.json"

DIAL_NOTE = (
    "## What the Dial does\n\n"
    "- **Amount 0**: Helm is off\n"
    "- **Amount 1**: the strongest setting that still looks right for this idea\n"
    "- **Before** = your prompt, Dial off. **After** = same prompt and seed, your Amount\n\n"
    "Only the picture changes along one idea. Your prompt is not rewritten.")


def steps_note(g, title, pos, size, first, second="Move **Amount** on the orange Dial", third="**Run**, then compare Before | After"):
    return g.add("MarkdownNote", pos, size, title=title, colors=NOTE, text=(
        "# Start here\n\n"
        f"## 1. {first}\n\n"
        f"## 2. {second}\n\n"
        f"## 3. {third}"))


def models_note(g, pos, links, size=(300, 120)):
    return g.add("MarkdownNote", pos, size, title="Note: Models", colors=NOTE, text=(
        "## Models\n\nLeave these as they are. If one is missing, ComfyUI offers to download it.\n\n" + links))


def dial_group(g, x, y, axis, extra_note="", w=540, note_h=150):
    """Group 'Move the Dial' with the Dial node and the 'What the Dial does' note. Returns the Dial id."""
    g.group("2 · Move the Dial" if g.dial_step == 2 else f"{g.dial_step} · Move the Dial", x, y, w, 80 + 160 + 30 + note_h + 20, GROUP_ORANGE)
    d = g.add("HelmDial", (x + 20, y + 80), (w - 40, 160), [axis, 0.5], "Helm · Dial", ORANGE)
    g.add("MarkdownNote", (x + 20, y + 270), (w - 40, note_h), colors=NOTE, title="Note: What the Dial does",
          text=DIAL_NOTE + extra_note)
    return d


def template_1():
    g = Graph()
    g.dial_step = 2
    steps_note(g, "Note: Start here", (0, 0), (500, 150), "Write your prompt in the green box")
    g.group("1 · Write your prompt", 0, 190, 500, 520, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (20, 270), (460, 200),
               ["an industrial hall with tall windows and a single person standing in the middle, documentary photography"],
               "Your prompt (edit me)", GREEN)
    size = g.add("EmptySD3LatentImage", (20, 500), (220, 150), [1024, 1024, 1], "Image size")
    seed = g.add("PrimitiveInt", (260, 500), (220, 110), [11, "fixed"], "Seed (shared by both)")
    d = dial_group(g, 560, 0, "Essentials · Light angle (raking) (Z-Image)")
    g.group("Models (leave as they are)", 560, 580, 640, 360, GROUP_GREY)
    u, c, v, s = z_models(g, 580, 660)
    zero = g.add("ConditioningZeroOut", (580, 860), (300, 70), None, "Empty negative", collapsed=True)
    models_note(g, (900, 660), "- [Z-Image Turbo files on Hugging Face](https://huggingface.co/Comfy-Org/z_image_turbo)")
    g.link(c, "CLIP", pr, "clip")
    g.link(pr, "CONDITIONING", zero, "conditioning")
    g.link(s, "MODEL", d, "model")
    g.group("3 · Result: Before | After", 1260, 0, 700, 760, GROUP_BLUE)
    before_after_z(g, 1260, 0, s, d, pr, zero, size, v, seed)
    return g


def template_2():
    g = Graph()
    g.dial_step = 2
    steps_note(g, "Note: Start here", (0, 0), (520, 150), "Describe your video in the green box")
    g.group("1 · Describe your video", 0, 190, 520, 560, GROUP_GREEN)
    h3 = g.add("MiniMaxH3ImageToVideo", (20, 270), (480, 400),
               [("Slow dolly shot through a packed industrial hall full of machinery, crates and cables, "
                 "documentary look, low ambient hum."), 768, 448, 73],
               "Your prompt (edit me)", GREEN)
    noise = g.add("RandomNoise", (20, 690), (300, 100), [11, "fixed"], "Seed (shared by both)", collapsed=True)
    d = dial_group(g, 580, 0, "Essentials · Less clutter (MiniMax H3)", w=540, note_h=170,
                   extra_note="\n\nThe video renders twice (Before and After), so it takes about twice as long.")
    g.group("Models (leave as they are)", 580, 620, 640, 480, GROUP_GREY)
    u = g.add("UNETLoader", (600, 700), (340, 100), ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"],
              "MiniMax H3 model", collapsed=True)
    lo = g.add("LoraLoaderModelOnly", (600, 750), (340, 110),
               ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1], "Fast mode (turbo, 6 steps)", collapsed=True)
    cl = g.add("CLIPLoader", (600, 800), (340, 130), ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
               "Text encoder", collapsed=True)
    vv = g.add("VAELoader", (920, 700), (340, 90), ["minimax_h3_video_vae_fp16.safetensors"], "Video decoder (VAE)", collapsed=True)
    va = g.add("VAELoader", (920, 750), (340, 90), ["minimax_h3_audio_vae_fp32.safetensors"], "Audio decoder (VAE)", collapsed=True)
    ks = g.add("KSamplerSelect", (920, 800), (340, 70), ["res_multistep"], "Sampler", collapsed=True)
    sch = g.add("BasicScheduler", (600, 850), (340, 110), ["simple", 6, 1], "Steps (fast mode = 6)", collapsed=True)
    models_note(g, (600, 920), "- [MiniMax H3 files on Hugging Face](https://huggingface.co/Comfy-Org/MiniMax-H3)", (600, 110))
    g.link(u, "MODEL", lo, "model")
    g.link(cl, "CLIP", h3, "clip"); g.link(vv, "VAE", h3, "vae")
    g.link(lo, "MODEL", sch, "model")
    g.link(lo, "MODEL", d, "model")
    g.group("3 · Result: Before | After", 1280, 0, 700, 1000, GROUP_BLUE)
    sv = g.add("SaveVideo", (1300, 80), (660, 440), ["video/helm-before-after", "auto", "auto"],
               "Left: Before    Right: After")
    row = 560
    gb = g.add("BasicGuider", (1300, row), (240, 70), None, "Before (Dial off)", collapsed=True)
    sb = g.add("SamplerCustomAdvanced", (1620, row), (240, 150), None, "Render Before", collapsed=True)
    db = g.add("VAEDecode", (1300, row + 50), (180, 60), None, "Decode Before", collapsed=True)
    ga = g.add("BasicGuider", (1620, row + 50), (240, 70), None, "After (your Amount)", collapsed=True)
    sa = g.add("SamplerCustomAdvanced", (1300, row + 100), (240, 150), None, "Render After", collapsed=True)
    da = g.add("VAEDecode", (1620, row + 100), (180, 60), None, "Decode After", collapsed=True)
    au = g.add("VAEDecodeAudio", (1300, row + 150), (180, 60), None, "Decode sound (After)", collapsed=True)
    st = g.add("ImageStitch", (1620, row + 150), (300, 190), ["right", True, 0, "white"], "Before | After", collapsed=True)
    cv = g.add("CreateVideo", (1300, row + 200), (300, 150), [24, "auto", "sRGB", "none"], "Make the video", collapsed=True)
    g.link(lo, "MODEL", gb, "model"); g.link(h3, "positive", gb, "conditioning")
    g.link(d, "model", ga, "model"); g.link(h3, "positive", ga, "conditioning")
    for s_, gd in ((sb, gb), (sa, ga)):
        g.link(noise, "NOISE", s_, "noise"); g.link(gd, "GUIDER", s_, "guider")
        g.link(ks, "SAMPLER", s_, "sampler"); g.link(sch, "SIGMAS", s_, "sigmas")
        g.link(h3, "LATENT", s_, "latent_image")
    g.link(sb, "output", db, "samples"); g.link(vv, "VAE", db, "vae")
    g.link(sa, "output", da, "samples"); g.link(vv, "VAE", da, "vae")
    g.link(sa, "output", au, "samples"); g.link(va, "VAE", au, "vae")
    g.link(db, "IMAGE", st, "image1"); g.link(da, "IMAGE", st, "image2")
    g.link(st, "IMAGE", cv, "images"); g.link(au, "AUDIO", cv, "audio")
    g.link(cv, "VIDEO", sv, "video")
    return g


def template_3():
    g = Graph()
    g.dial_step = 3
    steps_note(g, "Note: Start here", (0, 0), (560, 170), "Describe the idea twice in the green box: with it, without it",
               "**Run**: your idea is saved and the Dial can move it",
               "Move **Amount**, **Run** again, compare")
    g.group("1 · Describe the idea", 0, 210, 560, 760, GROUP_GREEN)
    tiny = g.add("EmptySD3LatentImage", (20, 290), (250, 130), [512, 512, 1], "Test size (small is fine)")
    ma = g.add("HelmMakeAxis", (20, 450), (520, 490),
               ["a quiet street at dawn, thick fog, documentary photography",
                "a quiet street at dawn, clear air, documentary photography",
                "my_fog", 4, 0, "fixed", 1.0, -1],
               "Helm · Make Axis", ORANGE)
    g.group("2 · Your prompt", 620, 0, 560, 400, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (640, 80), (520, 160), ["a quiet street at dawn, documentary photography"],
               "Your prompt (edit me)", GREEN)
    size = g.add("EmptySD3LatentImage", (640, 270), (250, 110), [1024, 1024, 1], "Final image size")
    seed = g.add("PrimitiveInt", (910, 270), (250, 110), [11, "fixed"], "Seed (shared by both)")
    g.group("3 · Move the Dial", 620, 440, 560, 560, GROUP_ORANGE)
    d = g.add("HelmDial", (640, 520), (520, 160), ["Custom · my_fog", 0.5], "Helm · Dial", ORANGE)
    g.add("MarkdownNote", (640, 710), (520, 230), colors=NOTE, title="Note: About your axis", text=(
        "## About your axis\n\n"
        "- Saved as **my_fog** in your ComfyUI user/steer folder. Next time it is also in the Dial menu under Custom.\n"
        "- Its safe range is an **estimate**, not a measurement. Trust the Before | After picture.\n"
        "- Nothing changes? Make the two prompts differ more.\n"
        "- Picture falls apart? Lower Amount.\n"
        "- Works with image models that take plain text prompts (Z-Image shown)."))
    g.group("4 · Result: Before | After", 1240, 0, 700, 760, GROUP_BLUE)
    g.group("Models (leave as they are)", 1240, 800, 640, 360, GROUP_GREY)
    u, c, v, s = z_models(g, 1260, 880)
    zero = g.add("ConditioningZeroOut", (1260, 1080), (300, 70), None, "Empty negative", collapsed=True)
    models_note(g, (1580, 880), "- [Z-Image Turbo files on Hugging Face](https://huggingface.co/Comfy-Org/z_image_turbo)")
    g.link(c, "CLIP", pr, "clip"); g.link(pr, "CONDITIONING", zero, "conditioning")
    g.link(s, "MODEL", ma, "model"); g.link(c, "CLIP", ma, "clip"); g.link(tiny, "LATENT", ma, "latent")
    g.link(ma, "model", d, "model"); g.link(ma, "axis", d, "axis_from")
    before_after_z(g, 1240, 0, s, d, pr, zero, size, v, seed)
    return g


T4 = "Helm 4. Chain ideas (Axis \u2192 Apply).json"
T5 = "Helm 5. Shape and mix (Axis \u00d7 3 \u2192 Mix \u2192 Shape \u2192 Apply).json"
T6 = "Helm 6. Build a multipurpose axis from prompt pairs.json"

AXIS_DEFAULT_PAIRS = ("a quiet street at dawn, thick fog || a quiet street at dawn, clear air\n"
                      "a mountain lake, thick fog || a mountain lake, clear air\n"
                      "a market square, thick fog || a market square, clear air")


def axis_node(g, pos, idea, title="Helm \u00b7 Axis", size=(400, 180)):
    """Library mode: the idea menu is the only thing to set (the other widgets are hidden by the frontend)."""
    return g.add("HelmAxis", pos, size, ["library", idea, AXIS_DEFAULT_PAIRS, "my_axis", True, 1.0, 4, 0, "fixed", 1.0, -1],
                 title, PURPLE)


def apply_node(g, pos, size=(400, 102)):
    return g.add("HelmApplyNode", pos, size, [False, "push"], "Helm \u00b7 Apply", GREY)


def shared_left(g, steps, w=500):
    """The green 'write your prompt' group, as in templates 1 and 4 to 5. Returns (prompt, size, seed)."""
    steps_note(g, "Note: Start here", (0, 0), (w, 170), *steps)
    g.group("1 · Write your prompt", 0, 210, w, 520, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (20, 290), (w - 40, 200),
               ["an industrial hall with tall windows and a single person standing in the middle, documentary photography"],
               "Your prompt (edit me)", GREEN)
    half = (w - 60) // 2
    size = g.add("EmptySD3LatentImage", (20, 520), (half, 150), [1024, 1024, 1], "Image size")
    seed = g.add("PrimitiveInt", (40 + half, 520), (half, 110), [11, "fixed"], "Seed (shared by both)")
    return pr, size, seed


def template_4():
    """Axis -> Apply: the two-node minimal path."""
    g = Graph()
    pr, size, seed = shared_left(g, ("Write your prompt in the green box",
                                     "Pick an idea on the purple **Axis**; its output goes into the grey **Apply**",
                                     "**Run**, then compare Before | After"))
    g.group("2 · Pick an idea", 560, 0, 460, 300, GROUP_PURPLE)
    ax = axis_node(g, (580, 80), "Essentials \u00b7 Light angle (raking) (Z-Image)")
    g.group("3 · Apply it", 560, 340, 460, 310, GROUP_ORANGE)
    ap = apply_node(g, (580, 420))
    g.add("MarkdownNote", (580, 550), (420, 80), colors=NOTE, title="Note: Axis \u2192 Apply", text=(
        "## The minimum chain\n\n"
        "- **Axis** says which idea, **Apply** puts it in the model\n"
        "- No Shape needed: amount 0.5 on every step"))
    g.group("Models (leave as they are)", 560, 690, 640, 360, GROUP_GREY)
    u, c, v, s = z_models(g, 580, 770)
    zero = g.add("ConditioningZeroOut", (580, 970), (300, 70), None, "Empty negative", collapsed=True)
    models_note(g, (900, 770), "- [Z-Image Turbo files on Hugging Face](https://huggingface.co/Comfy-Org/z_image_turbo)")
    g.link(c, "CLIP", pr, "clip")
    g.link(pr, "CONDITIONING", zero, "conditioning")
    g.link(s, "MODEL", ap, "model")
    g.link(ax, "helm", ap, "helm")
    g.group("4 · Result: Before | After", 1080, 0, 700, 760, GROUP_BLUE)
    before_after_z(g, 1080, 0, s, ap, pr, zero, size, v, seed)
    return g


def template_5():
    """Axis x 3 -> Mix -> Shape -> Apply."""
    g = Graph()
    pr, size, seed = shared_left(g, ("Write your prompt in the green box",
                                     "Pick three ideas on the purple **Axis** nodes",
                                     "Move **amount** on the orange **Shape**, **Run**, compare"), w=440)
    g.group("2 · Pick three ideas", 480, 0, 380, 700, GROUP_PURPLE)
    a1 = axis_node(g, (500, 80), "Essentials \u00b7 Light angle (raking) (Z-Image)", "Helm \u00b7 Axis: light", (340, 180))
    a2 = axis_node(g, (500, 280), "Essentials \u00b7 Less clutter (Z-Image)", "Helm \u00b7 Axis: clutter", (340, 180))
    a3 = axis_node(g, (500, 480), "Essentials \u00b7 Aisle width (Z-Image)", "Helm \u00b7 Axis: aisle", (340, 180))
    g.group("3 · Mix, shape, apply", 900, 0, 380, 780, GROUP_ORANGE)
    mx = g.add("HelmMix", (920, 80), (340, 280), [True, 1, 1, 1, 1, ""], "Helm \u00b7 Mix (experimental)", BLUE)
    sh = g.add("HelmShape", (920, 380), (340, 150), [0.5, "all steps", 0.5, "constant", 0.0, "linear",
                                                    "0:0.2, 0.5:0.8, 1:0.4"], "Helm \u00b7 Shape (experimental)", ORANGE)
    ap = apply_node(g, (920, 550), (340, 102))
    g.add("MarkdownNote", (920, 670), (340, 100), colors=NOTE, title="Note: Experimental", text=(
        "## Experimental\n\n"
        "**Shape** and **Mix** are new and not yet measured on a GPU. **Axis \u2192 Apply** (template 4) is the safe path."))
    g.group("Models (leave as they are)", 0, 770, 440, 330, GROUP_GREY)
    u, c, v, s = z_models(g, 20, 850)
    zero = g.add("ConditioningZeroOut", (20, 1050), (300, 70), None, "Empty negative", collapsed=True)
    models_note(g, (320, 850), "- [Z-Image Turbo files on Hugging Face](https://huggingface.co/Comfy-Org/z_image_turbo)", (110, 220))
    g.link(c, "CLIP", pr, "clip")
    g.link(pr, "CONDITIONING", zero, "conditioning")
    g.link(a1, "helm", mx, "helm_1"); g.link(a2, "helm", mx, "helm_2"); g.link(a3, "helm", mx, "helm_3")
    g.link(mx, "helm", sh, "helm")
    g.link(sh, "helm", ap, "helm")
    g.link(s, "MODEL", ap, "model")
    g.group("4 · Result: Before | After", 1320, 0, 640, 760, GROUP_BLUE)
    before_after_z(g, 1320, 0, s, ap, pr, zero, size, v, seed, pw=600)
    return g


def template_6():
    """An Axis built from prompt pairs, then Apply."""
    g = Graph()
    steps_note(g, "Note: Start here", (0, 0), (560, 190),
               "Write 3 pairs in the purple Axis: with the idea || without it, on different subjects",
               "**Run**: your idea is built, saved and applied",
               "Compare Before | After, then change the pairs")
    g.group("1 · Describe the idea", 0, 230, 560, 780, GROUP_PURPLE)
    tiny = g.add("EmptySD3LatentImage", (20, 310), (250, 130), [512, 512, 1], "Test size (small is fine)")
    ax = g.add("HelmAxis", (20, 470), (520, 520),
               ["prompts", "Essentials \u00b7 Light angle (raking) (Z-Image)", AXIS_DEFAULT_PAIRS, "my_fog", True, 1.0, 4, 0,
                "fixed", 1.0, -1], "Helm \u00b7 Axis: build from pairs", PURPLE)
    g.group("2 · Your prompt", 620, 0, 560, 400, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (640, 80), (520, 160), ["a quiet street at dawn, documentary photography"],
               "Your prompt (edit me)", GREEN)
    size = g.add("EmptySD3LatentImage", (640, 270), (250, 110), [1024, 1024, 1], "Final image size")
    seed = g.add("PrimitiveInt", (910, 270), (250, 110), [11, "fixed"], "Seed (shared by both)")
    g.group("3 · Apply it", 620, 440, 560, 430, GROUP_ORANGE)
    ap = apply_node(g, (640, 520), (520, 102))
    g.add("MarkdownNote", (640, 650), (520, 200), colors=NOTE, title="Note: About your axis", text=(
        "## About your axis\n\n"
        "- Saved as **my_fog** in your ComfyUI user/steer folder. Next time it is also in the Axis menu under Custom.\n"
        "- Its safe range is an **estimate**. Trust the Before | After picture.\n"
        "- **debias** (experimental) removes the colour tint a prompt change tends to bring.\n"
        "- Nothing changes? Make the two sides of each pair differ more."))
    g.group("4 · Result: Before | After", 1240, 0, 700, 760, GROUP_BLUE)
    g.group("Models (leave as they are)", 1240, 800, 640, 360, GROUP_GREY)
    u, c, v, s = z_models(g, 1260, 880)
    zero = g.add("ConditioningZeroOut", (1260, 1080), (300, 70), None, "Empty negative", collapsed=True)
    models_note(g, (1580, 880), "- [Z-Image Turbo files on Hugging Face](https://huggingface.co/Comfy-Org/z_image_turbo)")
    g.link(c, "CLIP", pr, "clip"); g.link(pr, "CONDITIONING", zero, "conditioning")
    g.link(s, "MODEL", ax, "model"); g.link(c, "CLIP", ax, "clip"); g.link(tiny, "LATENT", ax, "latent")
    g.link(s, "MODEL", ap, "model"); g.link(ax, "helm", ap, "helm")
    before_after_z(g, 1240, 0, s, ap, pr, zero, size, v, seed)
    return g


if __name__ == "__main__":
    # ComfyUI always fits a template to the window when it opens, and below about 60% zoom it draws nodes as
    # empty boxes. So each graph is kept compact (about 2000 x 1000 units) and the plumbing is collapsed.
    template_1().dump(os.path.join(OUT, T1))
    template_2().dump(os.path.join(OUT, T2))
    template_3().dump(os.path.join(OUT, T3))
    template_4().dump(os.path.join(OUT, T4))
    template_5().dump(os.path.join(OUT, T5))
    template_6().dump(os.path.join(OUT, T6))
    print("wrote 6 templates to", OUT)
