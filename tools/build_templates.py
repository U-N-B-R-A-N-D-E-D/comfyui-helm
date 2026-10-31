"""Regenerates the three starter templates in example_workflows/ (flat graphs, no subgraphs).

    python tools/build_templates.py

The templates are plain ComfyUI "UI format" JSON. They are generated so that the three graphs stay
consistent (same node sizes, colors, group titles) and so that every link is checked once, here and again
by tests/test_templates.py. Only core ComfyUI nodes and Helm nodes are used.
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
    "Note": ([], []),
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
    "HelmMakeAxis": ([("model", "MODEL", 0), ("clip", "CLIP", 0), ("latent", "LATENT", 0), ("with_idea", "STRING", 1),
                      ("without_idea", "STRING", 1), ("name", "STRING", 1), ("steps", "INT", 1),
                      ("seed", "INT", 1, 1), ("cfg", "FLOAT", 1, 1), ("block", "INT", 1, 1)],
                     [("model", "MODEL"), ("axis", "STRING")]),
}

# colors (title, body)
GREEN = ("#232", "#353")      # things you edit
ORANGE = ("#6b3300", "#a85200")   # the Helm control
NOTE = ("#432", "#653")
GROUP_BLUE = "#3f789e"
GROUP_ORANGE = "#b06634"
GROUP_GREEN = "#3f8e5e"
GROUP_GREY = "#8A8A8A"

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

    def add(self, typ, pos, size, widgets=None, title=None, colors=None, models=None, text=None):
        self.nid += 1
        ins, outs = S[typ]
        node = {
            "id": self.nid, "type": typ, "pos": list(pos), "size": list(size), "flags": {}, "order": self.nid,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outs],
            "properties": {"Node name for S&R": typ},
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
        if typ in ("HelmDial", "HelmMakeAxis"):
            node["properties"]["cnr_id"] = "comfyui-helm"
        if models:
            node["properties"]["models"] = [{"name": n, "url": u, "directory": d} for n, u, d in models]
        if text is not None:
            node["properties"]["text"] = text
            node["widgets_values"] = [text]
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
        self.gid += 1
        self.groups.append({"id": self.gid, "title": title, "bounding": [x, y, w, h], "color": color,
                            "font_size": 24, "flags": {}})

    def dump(self, path):
        doc = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(path))), "revision": 0,
               "last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes, "links": self.links,
               "groups": self.groups, "config": {}, "extra": {"ds": {"scale": 0.5, "offset": [60, 40]}},
               "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")


def z_models(g, x, y):
    u = g.add("UNETLoader", (x, y), (320, 100), [Z_MODELS["UNETLoader"][0], "default"], "Z-Image model",
              models=[Z_MODELS["UNETLoader"]])
    c = g.add("CLIPLoader", (x, y + 130), (320, 130), [Z_MODELS["CLIPLoader"][0], "lumina2", "default"],
              "Text encoder", models=[Z_MODELS["CLIPLoader"]])
    v = g.add("VAELoader", (x, y + 290), (320, 90), [Z_MODELS["VAELoader"][0]], "Image decoder (VAE)",
              models=[Z_MODELS["VAELoader"]])
    s = g.add("ModelSamplingAuraFlow", (x + 360, y), (300, 100), [3], "Sampling shift (leave as is)")
    g.link(u, "MODEL", s, "model")
    return u, c, v, s


def before_after_z(g, x, y, model_before, model_after, pos, neg, latent, vae, seed):
    """Two KSamplers with the same seed. Before = model straight from the loader chain (Dial off)."""
    kb = g.add("KSampler", (x, y), (320, 360), [0, "fixed", 8, 1, "res_multistep", "simple", 1],
               "Before (Dial off = Amount 0)")
    ka = g.add("KSampler", (x, y + 420), (320, 360), [0, "fixed", 8, 1, "res_multistep", "simple", 1],
               "After (your Amount)")
    for k, m in ((kb, model_before), (ka, model_after)):
        g.link(m, "MODEL", k, "model")
        g.link(pos, "CONDITIONING", k, "positive")
        g.link(neg, "CONDITIONING", k, "negative")
        g.link(latent, "LATENT", k, "latent_image")
        g.link(seed, "INT", k, "seed")
    db = g.add("VAEDecode", (x + 360, y), (200, 60), None, "Decode Before")
    da = g.add("VAEDecode", (x + 360, y + 420), (200, 60), None, "Decode After")
    g.link(kb, "LATENT", db, "samples"); g.link(vae, "VAE", db, "vae")
    g.link(ka, "LATENT", da, "samples"); g.link(vae, "VAE", da, "vae")
    st = g.add("ImageStitch", (x + 600, y), (300, 190), ["right", True, 0, "white"], "Before | After (side by side)")
    g.link(db, "IMAGE", st, "image1"); g.link(da, "IMAGE", st, "image2")
    pv = g.add("PreviewImage", (x + 600, y + 240), (640, 560), None, "Left: Before    Right: After")
    g.link(st, "IMAGE", pv, "images")
    sv = g.add("SaveImage", (x + 940, y), (300, 200), ["helm/after"], "Save the After image")
    g.link(da, "IMAGE", sv, "images")
    return kb, ka


NOTE_STEPS = "1) write your prompt\n2) move Amount\n3) Run"


def template_1():
    g = Graph()
    # 1 prompt
    g.group("1 · Write your prompt", 0, 0, 600, 600, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (20, 60), (560, 240),
               ["the same room packed with server racks, cables and crates, documentary photography"],
               "Your prompt (edit me)", GREEN)
    size = g.add("EmptySD3LatentImage", (20, 340), (300, 150), [1024, 1024, 1], "Image size")
    seed = g.add("PrimitiveInt", (340, 340), (240, 100), [11, "fixed"], "Seed (same for Before and After)")
    # 2 models
    g.group("2 · Models (leave as they are)", 0, 660, 720, 460, GROUP_GREY)
    u, c, v, s = z_models(g, 20, 720)
    zero = g.add("ConditioningZeroOut", (380, 860), (300, 70), None, "Empty negative prompt")
    g.link(c, "CLIP", pr, "clip")
    g.link(pr, "CONDITIONING", zero, "conditioning")
    # 3 dial
    g.group("3 · The Dial: move Amount", 780, 0, 560, 640, GROUP_ORANGE)
    n1 = g.add("Note", (800, 60), (520, 170), text=NOTE_STEPS, title="Helm 1 · Start here (Z-Image)", colors=NOTE)
    d = g.add("HelmDial", (800, 270), (520, 170), ["Essentials · Less clutter (Z-Image)", 0.5],
              "Helm · Dial", ORANGE)
    n2 = g.add("Note", (800, 470), (520, 140), colors=NOTE, title="What the Dial does", text=(
        "Amount 0 = Helm off. Amount 1 = the strongest setting that still looks right for this idea.\n"
        "Before = your prompt with the Dial off. After = the same prompt and seed with your Amount.\n"
        "Only the picture changes along one idea; your prompt is not rewritten."))
    g.link(s, "MODEL", d, "model")
    # 4 result
    g.group("4 · Result: Before | After", 1400, 0, 1280, 900, GROUP_BLUE)
    before_after_z(g, 1420, 60, s, d, pr, zero, size, v, seed)
    return g


def template_2():
    g = Graph()
    g.group("1 · Describe your video", 0, 0, 640, 660, GROUP_GREEN)
    h3 = g.add("MiniMaxH3ImageToVideo", (20, 60), (600, 420),
               [("Slow dolly shot through a packed industrial hall full of machinery, crates and cables, "
                 "documentary look, low ambient hum."), 768, 448, 73],
               "Your prompt (edit me)", GREEN)
    noise = g.add("RandomNoise", (20, 520), (300, 100), [11, "fixed"], "Seed (same for Before and After)")
    g.group("2 · Models (leave as they are)", 0, 720, 780, 560, GROUP_GREY)
    u = g.add("UNETLoader", (20, 780), (340, 100), ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"],
              "MiniMax H3 model")
    lo = g.add("LoraLoaderModelOnly", (20, 910), (340, 110),
               ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1], "Fast mode (turbo, 6 steps)")
    cl = g.add("CLIPLoader", (400, 780), (340, 130), ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
               "Text encoder")
    vv = g.add("VAELoader", (400, 940), (340, 90), ["minimax_h3_video_vae_fp16.safetensors"], "Video decoder (VAE)")
    va = g.add("VAELoader", (400, 1060), (340, 90), ["minimax_h3_audio_vae_fp32.safetensors"], "Audio decoder (VAE)")
    ks = g.add("KSamplerSelect", (20, 1060), (340, 70), ["res_multistep"], "Sampler")
    sch = g.add("BasicScheduler", (20, 1160), (340, 110), ["simple", 6, 1], "Steps (fast mode = 6)")
    g.link(u, "MODEL", lo, "model")
    g.link(cl, "CLIP", h3, "clip"); g.link(vv, "VAE", h3, "vae")
    g.link(lo, "MODEL", sch, "model")
    g.group("3 · The Dial: move Amount", 840, 0, 560, 660, GROUP_ORANGE)
    g.add("Note", (860, 60), (520, 170), text=NOTE_STEPS, title="Helm 2 · Video (MiniMax H3)", colors=NOTE)
    d = g.add("HelmDial", (860, 270), (520, 170), ["Essentials · Less clutter (MiniMax H3)", 0.5],
              "Helm · Dial", ORANGE)
    g.add("Note", (860, 470), (520, 160), colors=NOTE, title="What the Dial does", text=(
        "Amount 0 = Helm off. Amount 1 = the strongest setting that still looks right for this idea.\n"
        "This renders the video twice with the same seed, so it takes about twice as long: Before (Dial off) and After (your Amount)."))
    g.link(lo, "MODEL", d, "model")
    g.group("4 · Result: Before | After", 1460, 0, 1500, 1280, GROUP_BLUE)
    gb = g.add("BasicGuider", (1480, 60), (240, 70), None, "Before (Dial off)")
    ga = g.add("BasicGuider", (1480, 520), (240, 70), None, "After (your Amount)")
    g.link(lo, "MODEL", gb, "model"); g.link(h3, "positive", gb, "conditioning")
    g.link(d, "model", ga, "model"); g.link(h3, "positive", ga, "conditioning")
    sb = g.add("SamplerCustomAdvanced", (1760, 60), (260, 150), None, "Render Before")
    sa = g.add("SamplerCustomAdvanced", (1760, 520), (260, 150), None, "Render After")
    for s_, gd in ((sb, gb), (sa, ga)):
        g.link(noise, "NOISE", s_, "noise"); g.link(gd, "GUIDER", s_, "guider")
        g.link(ks, "SAMPLER", s_, "sampler"); g.link(sch, "SIGMAS", s_, "sigmas")
        g.link(h3, "LATENT", s_, "latent_image")
    db = g.add("VAEDecode", (2060, 60), (200, 60), None, "Decode Before")
    da = g.add("VAEDecode", (2060, 520), (200, 60), None, "Decode After")
    au = g.add("VAEDecodeAudio", (2060, 640), (200, 60), None, "Decode sound (After)")
    g.link(sb, "output", db, "samples"); g.link(vv, "VAE", db, "vae")
    g.link(sa, "output", da, "samples"); g.link(vv, "VAE", da, "vae")
    g.link(sa, "output", au, "samples"); g.link(va, "VAE", au, "vae")
    st = g.add("ImageStitch", (2300, 60), (300, 190), ["right", True, 0, "white"], "Before | After (side by side)")
    g.link(db, "IMAGE", st, "image1"); g.link(da, "IMAGE", st, "image2")
    cv = g.add("CreateVideo", (2300, 300), (300, 150), [24, "auto", "sRGB", "none"], "Make the video")
    g.link(st, "IMAGE", cv, "images"); g.link(au, "AUDIO", cv, "audio")
    sv = g.add("SaveVideo", (2300, 500), (400, 160), ["video/helm-before-after", "auto", "auto"],
               "Left: Before    Right: After")
    g.link(cv, "VIDEO", sv, "video")
    return g


def template_3():
    g = Graph()
    g.group("1 · Your prompt", 0, 0, 600, 480, GROUP_GREEN)
    pr = g.add("CLIPTextEncode", (20, 60), (560, 200), ["a quiet street at dawn, documentary photography"],
               "Your prompt (edit me)", GREEN)
    size = g.add("EmptySD3LatentImage", (20, 300), (300, 150), [1024, 1024, 1], "Final image size")
    seed = g.add("PrimitiveInt", (340, 300), (240, 100), [11, "fixed"], "Seed (same for Before and After)")
    g.group("2 · Models (leave as they are)", 0, 540, 720, 460, GROUP_GREY)
    u, c, v, s = z_models(g, 20, 600)
    zero = g.add("ConditioningZeroOut", (380, 740), (300, 70), None, "Empty negative prompt")
    g.link(c, "CLIP", pr, "clip"); g.link(pr, "CONDITIONING", zero, "conditioning")
    g.group("3 · Describe the idea", 780, 0, 640, 1000, GROUP_GREEN)
    g.add("Note", (800, 60), (600, 150), text=(
        "1) type the idea twice: with it, and without it (change only the idea)\n"
        "2) Run: Make Axis saves your axis and the Dial uses it straight away\n"
        "3) move Amount, Run again, compare Before | After"),
        title="Helm 3 · Make your own axis", colors=NOTE)
    tiny = g.add("EmptySD3LatentImage", (800, 250), (300, 150), [512, 512, 1], "Test size (small is fine)")
    ma = g.add("HelmMakeAxis", (800, 440), (600, 520),
               ["a quiet street at dawn, thick fog, documentary photography",
                "a quiet street at dawn, clear air, documentary photography",
                "my_fog", 4, 0, "fixed", 1.0, -1],
               "Helm · Make Axis", ORANGE)
    g.link(s, "MODEL", ma, "model"); g.link(c, "CLIP", ma, "clip"); g.link(tiny, "LATENT", ma, "latent")
    g.group("4 · The Dial", 1480, 0, 560, 640, GROUP_ORANGE)
    d = g.add("HelmDial", (1500, 60), (520, 170), ["Custom · my_fog", 0.5], "Helm · Dial", ORANGE)
    g.add("Note", (1500, 270), (520, 330), colors=NOTE, title="About your axis", text=(
        "The axis is saved as my_fog in your ComfyUI user/steer folder. Next time it also appears in the Dial menu under Custom.\n"
        "Its safe range is an ESTIMATE, not a measurement: the Essentials in the other templates were checked against a benchmark, yours was not. "
        "Trust the Before | After picture.\n"
        "If nothing changes, make the two prompts differ more. If the picture falls apart, lower Amount.\n"
        "Works with image models that take plain text prompts (Z-Image shown)."))
    g.link(ma, "model", d, "model"); g.link(ma, "axis", d, "axis_from")
    g.group("5 · Result: Before | After", 2100, 0, 1280, 900, GROUP_BLUE)
    before_after_z(g, 2120, 60, s, d, pr, zero, size, v, seed)
    return g


if __name__ == "__main__":
    template_1().dump(os.path.join(OUT, "helm_1_start_here.json"))
    template_2().dump(os.path.join(OUT, "helm_2_video_h3.json"))
    template_3().dump(os.path.join(OUT, "helm_3_make_your_own_axis.json"))
    print("wrote 3 templates to", OUT)
