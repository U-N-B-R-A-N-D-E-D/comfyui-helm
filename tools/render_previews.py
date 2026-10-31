"""Draws the preview images. No GPU: it only re-labels and re-crops frames that are already in the repo.

    python tools/render_previews.py     (needs Pillow)

1. web/previews/*.webp: real Helm renders, Before on the left and After on the right. The picture is the top
   FRAME_H rows; the strip underneath is only a label. The strip is repainted with plain words ("Before" /
   "After (Amount 1.0)"). The technical numbers (safe_max, strength) live in docs/AXES.md, not on the image.
2. example_workflows/<template name>.jpg: the template browser's card thumbnail. ComfyUI looks for
   "<template name>.jpg" next to the workflow, and the card is square (the official templates use 400x400).
   Templates 1 and 2 reuse the frames from step 1; template 3 is a layout drawing, not a render, because its
   axis is made on the spot.

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


def save_jpg(img, name):
    img.save(os.path.join(WF, name + ".jpg"), "JPEG", quality=88, optimize=True, progressive=True)


if __name__ == "__main__":
    for name, after in PREVIEWS.items():
        relabel(name, after)
    for name, (src, after) in THUMBS.items():
        save_jpg(thumb_before_after(src, after), name)
    save_jpg(thumb_make_axis(), "Helm 3. Make your own axis")
    print(f"wrote {len(PREVIEWS)} labelled previews and {len(THUMBS) + 1} template thumbnails")
