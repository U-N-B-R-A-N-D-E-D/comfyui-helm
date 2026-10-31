"""Draws a schematic preview (boxes for groups and nodes) for a starter template.

Only helm_3_make_your_own_axis uses it: its axis is made on the spot, so there is no stored render to show.
Templates 1 and 2 use real before/after frames from our own Helm measurement renders.

    python tools/render_previews.py     (needs Pillow)

It is a layout drawing generated from the template JSON, not a screenshot of ComfyUI.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, "example_workflows")
NAMES = {
    "helm_1_start_here": ("Helm 1 · Start here (Z-Image)", "one Dial, one idea. Before | After in one Run"),
    "helm_2_video_h3": ("Helm 2 · Video (MiniMax H3)", "the same Dial on video. Before | After"),
    "helm_3_make_your_own_axis": ("Helm 3 · Make your own axis", "two prompts in, a new Dial idea out"),
}


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(stem):
    d = json.load(open(os.path.join(WF, stem + ".json"), encoding="utf-8"))
    boxes = [g["bounding"] for g in d["groups"]] + [n["pos"] + n["size"] for n in d["nodes"]]
    x0 = min(b[0] for b in boxes) - 30; y0 = min(b[1] for b in boxes) - 60
    x1 = max(b[0] + b[2] for b in boxes) + 30; y1 = max(b[1] + b[3] for b in boxes) + 30
    W = 1280; s = W / (x1 - x0); H = int((y1 - y0) * s) + 150
    im = Image.new("RGB", (W, H), "#121214"); dr = ImageDraw.Draw(im)
    title, sub = NAMES[stem]
    dr.text((30, 22), title, fill="#f2f2f2", font=font(40)); dr.text((30, 74), sub, fill="#9a9a9f", font=font(24))
    T = lambda x, y: ((x - x0) * s, (y - y0) * s + 120)
    for g in d["groups"]:
        a, b = T(g["bounding"][0], g["bounding"][1]); c, e = T(g["bounding"][0] + g["bounding"][2], g["bounding"][1] + g["bounding"][3])
        dr.rectangle([a, b, c, e], outline=g["color"], width=3)
        dr.text((a + 8, b + 4), g["title"], fill=g["color"], font=font(17))
    for n in d["nodes"]:
        a, b = T(*n["pos"]); c, e = T(n["pos"][0] + n["size"][0], n["pos"][1] + n["size"][1])
        dr.rectangle([a, b, c, e], fill=n.get("bgcolor", "#353535"), outline="#777", width=1)
        dr.rectangle([a, b, c, b + 18], fill=n.get("color", "#222"))
        dr.text((a + 4, b + 1), (n.get("title") or n["type"])[:34], fill="#eee", font=font(12))
    dr.text((W - 190, H - 34), "comfyui-helm", fill="#9a9a9f", font=font(20))
    im.save(os.path.join(WF, stem + ".png"), optimize=True)


render("helm_3_make_your_own_axis")
print("wrote 1 preview")
