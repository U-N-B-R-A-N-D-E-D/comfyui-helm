"""Draws the preview images. No GPU: it only re-labels and re-crops frames that are already in the repo.

    python tools/render_previews.py     (needs Pillow; re-encodes the stored previews, so run it only when they change)
    python tools/render_previews.py --chain-only     (redraws only the drawn thumbnails of templates 3 to 6)

1. web/previews/*.webp: real Helm renders, Before on the left and After on the right. The picture is the top
   FRAME_H rows; the strip underneath is only a label. The strip is repainted with plain words ("Before" /
   "After (Amount 1.0)"). The technical numbers (safe_max, strength) live in docs/AXES.md, not on the image.
2. example_workflows/<template name>.jpg: the template browser's card thumbnail. ComfyUI looks for
   "<template name>.jpg" next to the workflow, and the card is square (the official templates use 400x400).
   Templates 1 and 2 reuse the frames from step 1; templates 3 to 6 are layout drawings, not renders: template 3's
   axis is made on the spot, and templates 4 to 6 teach a chain of nodes whose result depends on the idea you pick
   (the Axis -> Apply default, amount 0.5, was never rendered as a Before | After pair, so no frames are borrowed).

Output carries no metadata (no EXIF, no profile, no comments).
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREV = os.path.join(ROOT, "web", "previews")
WF = os.path.join(ROOT, "example_workflows")
FRAME_H = 294          # picture rows in a stored preview (the label strip is below)
STRIP_BG = (16, 16, 18)
LABEL_FG = (236, 236, 238)
THUMB = 512

# preview file -> (right-hand label)
PREVIEWS = {
    "z_raking_light_angle.webp": "After (Amount 1.0)",
    "z_overcast_vs_direct.webp": "After (Amount 1.0)",
    "z_aisle_width.webp": "After (Amount 1.0)",
    "z_corridor_orientation.webp": "After (Amount 1.0)",
    "density.webp": "After (Amount 0.67)",
}
# template name -> (preview its frames come from, right-hand label)
THUMBS = {
    "Helm 1. Start here (Z-Image)": ("z_raking_light_angle.webp", "After (Amount 1.0)"),
    "Helm 2. Video (MiniMax H3)": ("density.webp", "After (Amount 0.67)"),
}


def font(size, bold=False):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frames_of(path):
    im = Image.open(path).convert("RGB")
    w, _ = im.size
    pic = im.crop((0, 0, w, FRAME_H))
    return pic.crop((0, 0, w // 2, FRAME_H)), pic.crop((w // 2, 0, w, FRAME_H))


def relabel(name, after_text):
    path = os.path.join(PREV, name)
    before, after = frames_of(path)
    w = before.width * 2
    out = Image.new("RGB", (w, FRAME_H + 26), STRIP_BG)
    out.paste(before, (0, 0))
    out.paste(after, (before.width, 0))
    d = ImageDraw.Draw(out)
    f = font(15)
    for x, text in ((8, "Before"), (before.width + 8, after_text)):
        d.text((x, FRAME_H + 13), text, fill=LABEL_FG, font=f, anchor="lm")
    out.save(path, "WEBP", quality=88, method=6, exif=b"", icc_profile=None)


def pill(d, x, y, text, f, anchor="l"):
    w = d.textlength(text, font=f) + 28
    x0 = x if anchor == "l" else x - w
    d.rounded_rectangle([x0, y, x0 + w, y + 40], radius=20, fill=(0, 0, 0, 170))
    d.text((x0 + 14, y + 20), text, fill=(255, 255, 255), font=f, anchor="lm")


def thumb_before_after(src, after_text):
    before, after = frames_of(os.path.join(PREV, src))
    half = THUMB // 2
    scale = THUMB / FRAME_H
    out = Image.new("RGB", (THUMB, THUMB))
    for i, fr in enumerate((before, after)):
        big = fr.resize((round(fr.width * scale), THUMB), Image.LANCZOS)
        left = (big.width - half) // 2
        out.paste(big.crop((left, 0, left + half, THUMB)), (i * half, 0))
    d = ImageDraw.Draw(out, "RGBA")
    d.rectangle([half - 1, 0, half, THUMB], fill=(255, 255, 255, 235))
    f = font(22, True)
    pill(d, 14, THUMB - 56, "Before", f, "l")
    pill(d, THUMB - 14, THUMB - 56, after_text, f, "r")
    return out


def thumb_make_axis():
    out = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    d = ImageDraw.Draw(out)
    f, fb, fs = font(24, True), font(30, True), font(20)

    def card(box, fill, outline, title, sub):
        d.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        d.text((cx, cy - 12), title, fill=(255, 255, 255), font=f, anchor="mm")
        d.text((cx, cy + 18), sub, fill=(214, 224, 214), font=fs, anchor="mm")

    card((32, 34, 248, 154), (35, 51, 35), (63, 142, 94), "with the idea", "thick fog")
    card((264, 34, 480, 154), (35, 51, 35), (63, 142, 94), "without it", "clear air")
    for x in (140, 372):
        d.line([(x, 160), (x, 206)], fill=(150, 150, 155), width=4)
        d.polygon([(x - 10, 202), (x + 10, 202), (x, 220)], fill=(150, 150, 155))
    d.rounded_rectangle((32, 226, 480, 346), radius=18, fill=(168, 82, 0), outline=(176, 102, 52), width=3)
    d.text((256, 262), "Helm · Dial", fill=(255, 255, 255), font=fb, anchor="mm")
    d.rounded_rectangle((64, 300, 448, 312), radius=6, fill=(70, 34, 0))
    d.rounded_rectangle((64, 300, 270, 312), radius=6, fill=(255, 226, 194))
    d.ellipse((260, 293, 282, 319), fill=(255, 255, 255))
    d.text((256, 410), "My fog at 60%", fill=(255, 226, 194), font=fb, anchor="mm")
    d.text((256, 456), "one slider moves one idea", fill=(150, 150, 155), font=fs, anchor="mm")
    return out


# Helm role colors, the same as the template nodes and web/helm_axes.js: (fill, outline)
ROLE = {
    "axis": ((91, 58, 128), (126, 87, 163)),
    "shape": ((168, 82, 0), (176, 102, 52)),
    "mix": ((47, 95, 148), (63, 120, 158)),
    "apply": ((85, 85, 85), (138, 138, 138)),
    "edit": ((53, 83, 53), (63, 142, 94)),
}
ARROW = (150, 150, 155)


def role_card(d, box, role, title, sub=None, tf=None, sf=None):
    fill, outline = ROLE[role]
    d.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=3)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if sub:
        d.text((cx, cy - 11), title, fill=(255, 255, 255), font=tf or font(24, True), anchor="mm")
        d.text((cx, cy + 15), sub, fill=(236, 236, 238), font=sf or font(18), anchor="mm")
    else:
        d.text((cx, cy), title, fill=(255, 255, 255), font=tf or font(24, True), anchor="mm")


def down_arrow(d, x, y0, y1):
    d.line([(x, y0), (x, y1 - 12)], fill=ARROW, width=4)
    d.polygon([(x - 10, y1 - 14), (x + 10, y1 - 14), (x, y1)], fill=ARROW)


def thumb_chain():
    """Template 4: Axis -> Apply, the two-node minimal path."""
    out = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    d = ImageDraw.Draw(out)
    d.text((256, 52), "Axis, then Apply", fill=(255, 255, 255), font=font(36, True), anchor="mm")
    role_card(d, (76, 100, 436, 200), "axis", "Helm \u00b7 Axis", "Light angle (raking)")
    down_arrow(d, 256, 206, 262)
    role_card(d, (76, 268, 436, 358), "apply", "Helm \u00b7 Apply")
    down_arrow(d, 256, 364, 410)
    d.rounded_rectangle((136, 414, 376, 462), radius=14, fill=(40, 40, 44), outline=(90, 90, 96), width=2)
    d.text((256, 438), "your sampler", fill=(210, 210, 214), font=font(22), anchor="mm")
    d.text((256, 490), "two nodes, one idea", fill=ARROW, font=font(20), anchor="mm")
    return out


def thumb_mix():
    """Template 5: Axis x 3 -> Mix -> Shape -> Apply."""
    out = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    d = ImageDraw.Draw(out)
    f = font(20, True)
    for i, label in enumerate(("light", "clutter", "aisle")):
        x0 = 28 + i * 156
        role_card(d, (x0, 28, x0 + 140, 92), "axis", "Axis", label, f, font(17))
        d.line([(x0 + 70, 98), (x0 + 70, 118), (256, 130)], fill=ARROW, width=3)
    down_arrow(d, 256, 124, 150)
    role_card(d, (96, 150, 416, 214), "mix", "Helm \u00b7 Mix", "experimental", font(24, True), font(16))
    down_arrow(d, 256, 220, 262)
    role_card(d, (96, 262, 416, 326), "shape", "Helm \u00b7 Shape", "\u221240% (less), late steps", font(24, True), font(16))
    down_arrow(d, 256, 332, 374)
    role_card(d, (96, 374, 416, 430), "apply", "Helm \u00b7 Apply")
    d.text((256, 470), "three ideas, one slider", fill=ARROW, font=font(20), anchor="mm")
    return out


def thumb_pairs():
    """Template 6: prompt pairs -> Axis -> Apply."""
    out = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    d = ImageDraw.Draw(out)
    for i, (a, b) in enumerate((("street, fog", "street, clear"), ("lake, fog", "lake, clear"), ("square, fog", "square, clear"))):
        y0 = 24 + i * 56
        role_card(d, (36, y0, 244, y0 + 44), "edit", a, None, font(19, True))
        role_card(d, (268, y0, 476, y0 + 44), "edit", b, None, font(19, True))
        d.text((256, y0 + 22), "||", fill=(255, 255, 255), font=font(22, True), anchor="mm")
    down_arrow(d, 256, 196, 236)
    role_card(d, (76, 238, 436, 318), "axis", "Helm \u00b7 Axis", "built from your pairs")
    down_arrow(d, 256, 324, 362)
    role_card(d, (76, 364, 436, 424), "apply", "Helm \u00b7 Apply")
    d.text((256, 462), "three pairs, one reusable idea", fill=ARROW, font=font(20), anchor="mm")
    d.text((256, 490), "saved under Custom", fill=(120, 120, 126), font=font(17), anchor="mm")
    return out


def clear_of_badge(img, top=70):
    """The template card draws a 'Node graph' badge over the top-left of the thumbnail: keep the drawing below it."""
    scale = (THUMB - top) / THUMB
    small = img.resize((round(THUMB * scale), THUMB - top), Image.LANCZOS)
    out = Image.new("RGB", (THUMB, THUMB), (24, 24, 27))
    out.paste(small, ((THUMB - small.width) // 2, top))
    return out


def save_jpg(img, name):
    img.save(os.path.join(WF, name + ".jpg"), "JPEG", quality=88, optimize=True, progressive=True)


if __name__ == "__main__":
    import sys

    if "--chain-only" in sys.argv:
        # templates 3 to 6 are drawings: redraw them without re-encoding the stored previews or the Before | After thumbnails
        save_jpg(thumb_make_axis(), "Helm 3. Make your own axis")
        save_jpg(clear_of_badge(thumb_chain()), "Helm 4. Chain ideas (Axis \u2192 Apply)")
        save_jpg(clear_of_badge(thumb_mix()), "Helm 5. Shape and mix (Axis \u00d7 3 \u2192 Mix \u2192 Shape \u2192 Apply)")
        save_jpg(clear_of_badge(thumb_pairs()), "Helm 6. Build a multipurpose axis from prompt pairs")
        print("wrote 4 drawn template thumbnails")
        sys.exit(0)
    for name, after in PREVIEWS.items():
        relabel(name, after)
    for name, (src, after) in THUMBS.items():
        save_jpg(thumb_before_after(src, after), name)
    save_jpg(thumb_make_axis(), "Helm 3. Make your own axis")
    save_jpg(clear_of_badge(thumb_chain()), "Helm 4. Chain ideas (Axis \u2192 Apply)")
    save_jpg(clear_of_badge(thumb_mix()), "Helm 5. Shape and mix (Axis \u00d7 3 \u2192 Mix \u2192 Shape \u2192 Apply)")
    save_jpg(clear_of_badge(thumb_pairs()), "Helm 6. Build a multipurpose axis from prompt pairs")
    print(f"wrote {len(PREVIEWS)} labelled previews and {len(THUMBS) + 4} template thumbnails")
