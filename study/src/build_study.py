#!/usr/bin/env python3
"""Build the Helm study layer from the measured JSON artefacts.

Reads docs/data/*.json and the media archive, writes normalised CSVs,
metrics, charts, the axis atlas and study/README.md.

The JSON files are the source of truth; this directory is derived.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"
STUDY = ROOT / "study"
OUT_DATA = STUDY / "data"
OUT_RESULTS = STUDY / "results"
OUT_CHARTS = STUDY / "charts"
OUT_ATLAS = STUDY / "atlas"
OUT_THUMBS = OUT_ATLAS / "thumbs"

EVO = Path.home() / "HAL-archive/helm-videos/build/v38-intake/z19/2026-10-19/evo"
FONT_PATH = (
    Path.home()
    / "HAL-archive/helm-videos/source/vo-music/fonts/FragmentMono-Regular.ttf"
)

BG = "#fbfbfa"
INK = "#111111"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#4f46e5"
GREEN = "#16a34a"
RED = "#dc2626"
GRAY = "#9ca3af"

CATEGORY_COLORS = {
    "light": "#f59e0b",
    "camera": "#38bdf8",
    "material": "#a78bfa",
    "atmosphere": "#34d399",
    "geometry": "#fb7185",
}

CATEGORY = {
    "z_aisle_width": "geometry",
    "z_artificial_light_bursts": "light",
    "z_atmospheric_haze_density": "atmosphere",
    "z_backlight_rim": "light",
    "z_bw_film_stock": "material",
    "z_camera_motion_intensity": "camera",
    "z_corridor_orientation": "geometry",
    "z_figure_ground_separation": "camera",
    "z_film_grain": "material",
    "z_fluorescent_vs_tungsten": "light",
    "z_fog_density": "atmosphere",
    "z_foreground_occlusion": "camera",
    "z_framing_angle": "camera",
    "z_geometric_vs_organic": "geometry",
    "z_high_key_vs_low_key": "light",
    "z_horizon_height": "camera",
    "z_lens_distortion": "camera",
    "z_lens_flare_intensity": "light",
    "z_light_source_position": "light",
    "z_matte_vs_glossy": "material",
    "z_mixed_color_sources": "light",
    "z_motion_blur": "camera",
    "z_motion_blur_intensity": "camera",
    "z_night_practicals": "light",
    "z_overcast_vs_direct": "atmosphere",
    "z_perspective_convergence": "camera",
    "z_prism_effect_intensity": "light",
    "z_pro_mist_diffusion": "atmosphere",
    "z_raking_light_angle": "light",
    "z_refraction_focus": "material",
    "z_refraction_index_shift": "material",
    "z_repetition_vs_variety": "geometry",
    "z_saturated_vs_muted_stock": "material",
    "z_sensor_noise_iso": "material",
    "z_shading_intensity": "material",
    "z_shadow_depth": "geometry",
    "z_shadow_hardness": "geometry",
    "z_shadows_on_floor": "geometry",
    "z_snow_light": "light",
    "z_spatial_depth": "camera",
    "z_specular_reflection_density": "material",
    "z_specular_reflection_intensity": "material",
    "z_subject_scale": "camera",
    "z_sun_angle": "light",
    "z_sun_direction": "light",
    "z_sun_raking_angle": "light",
    "z_surface_reflection_intensity": "material",
    "z_symmetry": "camera",
    "z_telephoto_compression": "camera",
    "z_vanishing_point_position": "camera",
    "z_vignetting": "geometry",
    "z_volumetric_beams": "light",
    "z_volumetric_scattering": "atmosphere",
    "z_wet_vs_dry_reflectance": "material",
    "z_window_vs_practicals": "light",
    "z_lens_flare": "light",
    "z_atmospheric_haze": "atmosphere",
    "z_chromatic_aberration": "material",
    "z_depth_of_field": "camera",
    "z_floor_reflectivity": "material",
    "z_framing_focus": "camera",
    "z_ceiling_height": "geometry",
    "z_volumetric_beams_intensity": "light",
    "z_light_quality": "light",
    "z_twilight_haze": "atmosphere",
    "z_sunset_silhouette": "light",
    "z_light_source_intensity": "light",
    "z_color_temperature_gradient": "light",
    "z_depth_of_field_effect": "camera",
    "z_chromatic_aberration_shift": "material",
    "z_exposure_control": "light",
    "z_camera_motion": "camera",
    "z_light_path_reflection": "material",
    "z_golden_hour": "light",
    "z_color_temperature": "light",
    "z_ambient_light_source": "light",
    "z_moonlight_reflection": "light",
    "z_color_temp_shift": "light",
}

CATEGORY_ORDER = ["light", "camera", "material", "atmosphere", "geometry"]


def read_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def points(axis_data, mode):
    raw = axis_data.get(mode) or []
    return sorted(raw, key=lambda p: p["strength"])


def edge_ratio(point, base):
    if not base or base.get("edge") in (None, 0):
        return None
    return point["edge"] / base["edge"]


def wash_onset(axis_data):
    base = axis_data["baseline"]
    if not base:
        return None
    for point in points(axis_data, "push"):
        ratio = edge_ratio(point, base)
        if ratio is not None and ratio < 0.5:
            return point["strength"]
        if point["sat"] > 0.5:
            return point["strength"]
        if abs(point["lum"] - base["lum"]) > 0.2:
            return point["strength"]
    return None


def build_axis_rows(labels, axes78):
    by_name = {row["axis"]: row for row in axes78}
    rows = []
    for name, meta in sorted(by_name.items()):
        axis_data = labels["axes"].get(name)
        category = CATEGORY.get(name, "other")
        if not axis_data:
            rows.append(
                {
                    "axis": name,
                    "category": category,
                    "cos": fmt(meta.get("cos"), 4),
                    "norm": fmt(meta.get("norm"), 2),
                    "dosed": "no",
                    "label_push": "",
                    "strict_push": "",
                    "base_clip": "",
                    "base_edge": "",
                    "base_lum": "",
                    "base_sat": "",
                    "top_strength": "",
                    "top_clip": "",
                    "clip_delta": "",
                    "edge_ratio_top": "",
                    "lum_delta_max": "",
                    "sat_top": "",
                    "wash_onset": "",
                    "monotone": "",
                    "spearman": "",
                }
            )
            continue
        base = axis_data["baseline"]
        push = points(axis_data, "push")
        top = push[-1] if push else None
        edge_ratios = [edge_ratio(p, base) for p in push]
        lum_deltas = [abs(p["lum"] - base["lum"]) for p in push]
        clip_values = [p["clip"] for p in push]
        strengths = [p["strength"] for p in push]
        deltas = [
            clip_values[i + 1] - clip_values[i] for i in range(len(clip_values) - 1)
        ]
        monotone = bool(deltas) and (
            all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)
        )
        rows.append(
            {
                "axis": name,
                "category": category,
                "cos": fmt(meta.get("cos"), 4),
                "norm": fmt(meta.get("norm"), 2),
                "dosed": "yes" if meta.get("dosed") else "no",
                "label_push": axis_data["labels"]["push"],
                "strict_push": axis_data.get("strict_push", ""),
                "base_clip": fmt(base["clip"], 4),
                "base_edge": fmt(base["edge"], 5),
                "base_lum": fmt(base["lum"], 4),
                "base_sat": fmt(base["sat"], 4),
                "top_strength": fmt(top["strength"], 4) if top else "",
                "top_clip": fmt(top["clip"], 4) if top else "",
                "clip_delta": fmt(top["clip"] - base["clip"], 4) if top else "",
                "edge_ratio_top": fmt(edge_ratios[-1], 3) if edge_ratios else "",
                "lum_delta_max": fmt(max(lum_deltas), 3) if lum_deltas else "",
                "sat_top": fmt(top["sat"], 3) if top else "",
                "wash_onset": fmt(wash_onset(axis_data), 3),
                "monotone": "yes" if monotone else "no",
                "spearman": fmt(spearman(strengths, clip_values), 3),
            }
        )
    return rows


def build_long_rows(labels):
    rows = []
    for name, axis_data in sorted(labels["axes"].items()):
        base = axis_data["baseline"]
        rows.append(
            {
                "axis": name,
                "category": CATEGORY.get(name, "other"),
                "mode": "baseline",
                "strength": "0.0000",
                "lum": fmt(base["lum"], 5),
                "sat": fmt(base["sat"], 5),
                "edge": fmt(base["edge"], 6),
                "clip": fmt(base["clip"], 5),
                "label": "",
            }
        )
        for mode in ("push", "norm", "clamp"):
            label = axis_data["labels"].get(mode, "")
            for point in points(axis_data, mode):
                rows.append(
                    {
                        "axis": name,
                        "category": CATEGORY.get(name, "other"),
                        "mode": mode,
                        "strength": fmt(point["strength"], 4),
                        "lum": fmt(point["lum"], 5),
                        "sat": fmt(point["sat"], 5),
                        "edge": fmt(point["edge"], 6),
                        "clip": fmt(point["clip"], 5),
                        "label": label,
                    }
                )
    return rows


def build_window_rows(labels):
    rows = []
    for name, axis_data in sorted(labels["axes"].items()):
        base = axis_data["baseline"]
        for point in axis_data.get("window") or []:
            rows.append(
                {
                    "axis": name,
                    "category": CATEGORY.get(name, "other"),
                    "strength": fmt(point.get("strength"), 4),
                    "lum": fmt(point["lum"], 5),
                    "sat": fmt(point["sat"], 5),
                    "edge": fmt(point["edge"], 6),
                    "clip": fmt(point["clip"], 5),
                    "edge_ratio": fmt(edge_ratio(point, base), 3),
                }
            )
    return rows


def build_dose_rows(doses):
    rows = []
    for point in doses:
        rows.append(
            {
                "axis": point["axis"],
                "category": CATEGORY.get(point["axis"], "other"),
                "s_ref": fmt(point.get("s_ref"), 4),
                "strength": fmt(point.get("strength"), 4),
                "lum": fmt(point.get("lum"), 5),
                "sat": fmt(point.get("sat"), 5),
                "edge": fmt(point.get("edge"), 5),
                "dlum": fmt(point.get("dlum"), 5),
                "dsat": fmt(point.get("dsat"), 5),
                "dedge": fmt(point.get("dedge"), 5),
            }
        )
    return rows


def build_bench_rows(bench):
    results = bench["results"]
    rows = []
    for name, value in results.items():
        match = re.match(r"^(full|helm|mix)([\d.]+)?_s(\d+)$", name)
        if not match:
            continue
        family, level, seed = match.group(1), match.group(2), match.group(3)
        base = results.get(f"full_s{seed}")
        if not base:
            base = results["full_s11"]
        rows.append(
            {
                "variant": name,
                "family": family,
                "level": level or "0",
                "seed": seed,
                "lum": fmt(value["lum"], 4),
                "sat": fmt(value["sat"], 4),
                "edge": fmt(value["edge"], 5),
                "p_empty": fmt(value["p_empty"], 4),
                "p_packed": fmt(value["p_packed"], 4),
                "lum_ratio": fmt(value["lum"] / base["lum"], 3) if base["lum"] else "",
                "edge_ratio": fmt(value["edge"] / base["edge"], 3)
                if base["edge"]
                else "",
            }
        )
    return rows


def build_provenance_rows(lines):
    rows = []
    for line in lines:
        item = json.loads(line)
        rows.append(
            {
                "axis": item.get("axis", ""),
                "stage": item.get("stage", ""),
                "mode": item.get("mode", ""),
                "strength": fmt(item.get("strength"), 4),
                "status": item.get("status", ""),
                "output": item.get("output", ""),
                "sha256": item.get("sha256", ""),
                "graph_sha256": item.get("graph_sha256", ""),
                "job_id": item.get("job_id", ""),
            }
        )
    return rows


def survival_rows(axis_rows):
    grouped = defaultdict(list)
    for row in axis_rows:
        if row["label_push"] in ("holds", "washes"):
            grouped[row["category"]].append(row)
    rows = []
    for category in CATEGORY_ORDER:
        items = grouped.get(category, [])
        holds = sum(1 for r in items if r["label_push"] == "holds")
        strict = sum(1 for r in items if r["strict_push"] == "holds")
        clips = [float(r["clip_delta"]) for r in items if r["clip_delta"]]
        edges = [float(r["edge_ratio_top"]) for r in items if r["edge_ratio_top"]]
        rows.append(
            {
                "category": category,
                "labelled": len(items),
                "holds": holds,
                "washes": len(items) - holds,
                "hold_rate": fmt(pct(holds, len(items)), 1),
                "strict_holds": strict,
                "strict_rate": fmt(pct(strict, len(items)), 1),
                "median_clip_delta": fmt(float(np.median(clips)) if clips else None, 3),
                "median_edge_ratio": fmt(float(np.median(edges)) if edges else None, 3),
            }
        )
    return rows


def model_rows(calibration):
    measured = {m["match"][0]: m for m in calibration["models"]}
    h3 = measured["minimax"]
    z = measured["z_image"]
    ltx = measured["ltx"]
    rows = [
        {
            "model": "MiniMax H3",
            "module": h3["module"],
            "blocks": h3["n_blocks"],
            "dim": h3["dim"],
            "default_block": h3["default_block"],
            "band": "15-35",
            "strength": "0.01-0.30",
            "status": "measured",
        },
        {
            "model": "Z-Image / Lumina2",
            "module": z["module"],
            "blocks": z["n_blocks"],
            "dim": z["dim"],
            "default_block": z["default_block"],
            "band": "9-21",
            "strength": "0.50-3.00",
            "status": "measured",
        },
        {
            "model": "LTX-2.3",
            "module": ltx["module"],
            "blocks": "",
            "dim": "",
            "default_block": ltx.get("default_block", 20),
            "band": "",
            "strength": "",
            "status": "hook ready, untested",
        },
        {
            "model": "Krea 2",
            "module": "blocks",
            "blocks": "",
            "dim": "",
            "default_block": "",
            "band": "",
            "strength": "",
            "status": "wired, no calibration",
        },
        {
            "model": "Flux (DoubleStreamBlock)",
            "module": "double_blocks",
            "blocks": "",
            "dim": "",
            "default_block": "",
            "band": "",
            "strength": "",
            "status": "tuple slot declared, no dose",
        },
        {
            "model": "SD3",
            "module": "joint_blocks",
            "blocks": "",
            "dim": "",
            "default_block": "",
            "band": "",
            "strength": "",
            "status": "refused until measured",
        },
        {
            "model": "Qwen-Image",
            "module": "transformer_blocks",
            "blocks": "",
            "dim": "",
            "default_block": "",
            "band": "",
            "strength": "",
            "status": "refused until measured",
        },
    ]
    return rows


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def setup_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def save(fig, name):
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / name, dpi=170)
    plt.close(fig)


def chart_funnel(labels, axes78):
    stages = [
        ("Captured", len(axes78)),
        ("Dosed", sum(1 for r in axes78 if r.get("dosed"))),
        (
            "Labelled",
            sum(
                1
                for a in labels["axes"].values()
                if a["labels"]["push"] != "unmeasured"
            ),
        ),
        (
            "Holds (pre-registered)",
            sum(1 for a in labels["axes"].values() if a["labels"]["push"] == "holds"),
        ),
        (
            "Holds (strict ceiling)",
            sum(1 for a in labels["axes"].values() if a.get("strict_push") == "holds"),
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]
    colors = ["#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"]
    bars = ax.barh(names, values, color=colors, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=10,
            weight="bold",
        )
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("directions")
    ax.set_title("From 78 captured directions to 7 strict survivors")
    ax.grid(axis="y", visible=False)
    save(fig, "01_survival_funnel.png")


def chart_categories(survival):
    rows = [r for r in survival if r["labelled"] > 0]
    rows.sort(key=lambda r: (float(r["hold_rate"]), r["category"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))
    holds = [r["holds"] for r in rows]
    washes = [r["washes"] for r in rows]
    ax.barh(y, holds, color=GREEN, height=0.6, label="holds")
    ax.barh(y, washes, left=holds, color="#fecaca", height=0.6, label="washes")
    for i, row in enumerate(rows):
        ax.text(
            row["labelled"] + 0.4,
            i,
            f"{row['holds']}/{row['labelled']}  ({row['hold_rate']}%)",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['category']}" for r in rows])
    ax.set_xlabel("labelled axes")
    ax.set_title("Survival by axis family")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["labelled"] for r in rows) * 1.45)
    save(fig, "02_holds_by_category.png")


def chart_dose_curves(labels):
    wanted = [
        "z_aisle_width",
        "z_fog_density",
        "z_framing_angle",
        "z_specular_reflection_intensity",
        "z_high_key_vs_low_key",
        "z_shadow_hardness",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    palette = ["#4f46e5", "#0ea5e9", "#f59e0b", "#a855f7", "#16a34a", "#ef4444"]
    for color, name in zip(palette, wanted):
        data = labels["axes"].get(name)
        if not data:
            continue
        base = data["baseline"]
        push = points(data, "push")
        xs = [p["strength"] for p in push]
        ys = [p["clip"] - base["clip"] for p in push]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=color,
            label=name.replace("z_", ""),
        )
    ax.axhline(0, color=INK, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("strength")
    ax.set_ylabel("CLIP contrast, change vs baseline")
    ax.set_title("Dose response is not uniform across axes")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    save(fig, "03_dose_curves.png")


def chart_cos(labels, axes78):
    by_name = {r["axis"]: r for r in axes78}
    hold_x, wash_x, undosed_x = [], [], []
    for name, data in labels["axes"].items():
        cos = by_name.get(name, {}).get("cos")
        if cos is None:
            continue
        if data["labels"]["push"] == "holds":
            hold_x.append(cos)
        elif data["labels"]["push"] == "washes":
            wash_x.append(cos)
        else:
            undosed_x.append(cos)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(
        wash_x,
        rng.normal(0.0, 0.06, len(wash_x)),
        s=28,
        color=RED,
        alpha=0.65,
        label="washes",
    )
    ax.scatter(
        hold_x,
        rng.normal(1.0, 0.06, len(hold_x)),
        s=36,
        color=GREEN,
        alpha=0.9,
        label="holds",
    )
    if undosed_x:
        ax.scatter(
            undosed_x,
            rng.normal(-0.6, 0.04, len(undosed_x)),
            s=22,
            color=GRAY,
            alpha=0.7,
            label="unmeasured",
        )
    low_hold = min(hold_x) if hold_x else None
    high_wash = max(wash_x) if wash_x else None
    if low_hold is not None:
        ax.axvline(low_hold, color=GREEN, linestyle=":", linewidth=1)
    if high_wash is not None:
        ax.axvline(high_wash, color=RED, linestyle=":", linewidth=1)
    if low_hold is not None and high_wash is not None:
        ax.axvspan(
            min(low_hold, high_wash),
            max(low_hold, high_wash),
            color="#fef3c7",
            alpha=0.6,
        )
        ax.text(
            (low_hold + high_wash) / 2,
            1.45,
            "overlap",
            ha="center",
            fontsize=8,
            color="#92400e",
        )
    ax.set_yticks([-0.6, 0.0, 1.0])
    ax.set_yticklabels(["unmeasured", "washes", "holds"])
    ax.set_xlabel("cos(axis, mean activation)")
    ax.set_title("The mean-activation screen does not separate holds from washes")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(-0.9, 1.6)
    save(fig, "04_cos_vs_survival.png")


def chart_wash_map(axis_rows):
    rows = [r for r in axis_rows if r["label_push"] in ("holds", "washes")]
    rows.sort(
        key=lambda r: (
            CATEGORY_ORDER.index(r["category"])
            if r["category"] in CATEGORY_ORDER
            else 99,
            r["axis"],
        )
    )
    labels = [r["axis"].replace("z_", "") for r in rows]
    cols = ["low", "mid", "high"]
    matrix = []
    for row in rows:
        data = AXIS_LABELS["axes"][row["axis"]]
        base = data["baseline"]
        push = points(data, "push")
        values = []
        for point in push:
            ratio = edge_ratio(point, base)
            values.append(ratio if ratio is not None else np.nan)
        while len(values) < 3:
            values.append(np.nan)
        matrix.append(values[:3])
    matrix = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 11.5))
    cmap = LinearSegmentedColormap.from_list(
        "helm", ["#b91c1c", "#fca5a5", "#fde68a", "#86efac", "#15803d"]
    )
    im = ax.imshow(
        np.clip(matrix, 0, 1.6), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.6
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.1)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(CATEGORY_COLORS.get(row["category"], MUTED))
    for index, row in enumerate(rows):
        if index and rows[index - 1]["category"] != row["category"]:
            ax.axhline(index - 0.5, color=BG, linewidth=2.2)
    ax.set_title("Edge retained vs baseline (green = structure, red = wash)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("edge ratio")
    ax.grid(False)
    save(fig, "05_wash_map.png")


def chart_effect(axis_rows):
    rows = [r for r in axis_rows if r["label_push"] in ("holds", "washes")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, color, marker in (("holds", GREEN, "o"), ("washes", RED, "x")):
        xs = [
            float(r["edge_ratio_top"])
            for r in rows
            if r["label_push"] == label and r["edge_ratio_top"]
        ]
        ys = [
            float(r["clip_delta"])
            for r in rows
            if r["label_push"] == label and r["clip_delta"]
        ]
        ax.scatter(
            xs,
            ys,
            s=34 if label == "holds" else 26,
            color=color,
            marker=marker,
            alpha=0.8,
            label=label,
        )
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_xlabel("edge ratio at top dose")
    ax.set_ylabel("CLIP contrast change at top dose")
    ax.set_title("Survivors keep structure and move the CLIP contrast")
    ax.legend(loc="lower right")
    for row in sorted(
        rows, key=lambda r: abs(float(r["clip_delta"] or 0)), reverse=True
    )[:4]:
        if row["clip_delta"] and row["edge_ratio_top"]:
            ax.annotate(
                row["axis"].replace("z_", ""),
                (float(row["edge_ratio_top"]), float(row["clip_delta"])),
                fontsize=6.5,
                color=MUTED,
                xytext=(4, 4),
                textcoords="offset points",
            )
    save(fig, "06_effect_size.png")


def chart_bench(bench_rows):
    order = [
        "full_s11",
        "full_s23",
        "helm0.5_s11",
        "helm0.5_s23",
        "helm1.0_s11",
        "mix0.9_s11",
        "mix0.8_s11",
        "mix0.7_s11",
        "mix0.5_s11",
        "mix0.3_s11",
    ]
    by_name = {r["variant"]: r for r in bench_rows}
    variants = [v for v in order if v in by_name]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    y = np.arange(len(variants))
    lum = [float(by_name[v]["lum_ratio"] or 1) for v in variants]
    edge = [float(by_name[v]["edge_ratio"] or 1) for v in variants]
    ax1.barh(y - 0.18, lum, height=0.34, color="#f59e0b", label="luminance")
    ax1.barh(y + 0.18, edge, height=0.34, color="#6366f1", label="edge density")
    ax1.axvline(1.0, color=INK, linewidth=0.8, linestyle="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(variants, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("ratio vs full-room baseline")
    ax1.set_title("Helm grades; prompt mix steps")
    ax1.legend(loc="lower right", fontsize=8)
    p_empty = [float(by_name[v]["p_empty"]) for v in variants]
    ax2.barh(y, p_empty, height=0.5, color="#0ea5e9")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("P(empty)")
    ax2.set_title("Scene flip")
    ax2.invert_yaxis()
    save(fig, "07_prompt_mix.png")


def chart_provenance(prov_rows):
    stages = Counter(r["stage"] for r in prov_rows)
    modes = Counter(r["mode"] for r in prov_rows)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1, 1]}
    )
    stage_names = list(stages)
    stage_values = [stages[s] for s in stage_names]
    ax1.barh(stage_names, stage_values, color="#818cf8", height=0.55)
    for i, value in enumerate(stage_values):
        ax1.text(value + 2, i, str(value), va="center", fontsize=8)
    ax1.set_title("385 provenance jobs by stage")
    ax1.set_xlabel("jobs")
    ax1.grid(axis="y", visible=False)
    mode_names = list(modes)
    mode_values = [modes[m] for m in mode_names]
    ax2.barh(mode_names, mode_values, color="#34d399", height=0.55)
    for i, value in enumerate(mode_values):
        ax2.text(value + 2, i, str(value), va="center", fontsize=8)
    ax2.set_title("by injection mode")
    ax2.set_xlabel("jobs")
    ax2.grid(axis="y", visible=False)
    save(fig, "08_provenance.png")


def chart_model_matrix(rows):
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.axis("off")
    headers = ["model", "wired", "calibrated", "measured"]
    cells = []
    for row in rows:
        status = row["status"]
        measured = status == "measured"
        calibrated = status == "measured"
        wired = status != "refused until measured"
        cells.append(
            [
                row["model"],
                "yes" if wired else "no",
                "yes" if calibrated else "no",
                "yes" if measured else "no",
            ]
        )
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colWidths=[0.42, 0.16, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#eef2ff")
            cell.set_text_props(weight="bold")
        elif c > 0:
            value = cell.get_text().get_text()
            cell.set_facecolor("#dcfce7" if value == "yes" else "#f3f4f6")
    ax.set_title("Calibration coverage: two models measured, five not")
    save(fig, "09_model_matrix.png")


def make_thumb(axis, panels):
    width, height, header, gap = 260, 260, 24, 8
    canvas_w = width * 4 + gap * 5
    canvas_h = height + header + gap * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#0a0a0a")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(str(FONT_PATH), 14)
    except OSError:
        font = ImageFont.load_default()
    for index, (path, caption) in enumerate(panels):
        x = gap + index * (width + gap)
        if path and Path(path).exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((width, height), Image.LANCZOS)
            canvas.paste(image, (x + (width - image.width) // 2, header + gap))
        else:
            draw.rectangle(
                [x, header + gap, x + width, header + gap + height], fill="#1f2937"
            )
        draw.text((x + 4, 6), caption, fill="#d1d5db", font=font)
    target = OUT_THUMBS / f"{axis}.webp"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(target, "WEBP", quality=82, method=6)
    except (OSError, ValueError):
        canvas.save(target.with_suffix(".png"), "PNG")
        target = target.with_suffix(".png")
    return target


def strength_from_name(path):
    name = Path(path).name
    match = re.search(r"_(norm|clamp)_([0-9.]+)_", name) or re.search(
        r"_push_([0-9.]+)_win_", name
    )
    if not match:
        return None
    return float(match.group(2)) if match.lastindex == 2 else float(match.group(1))


def axis_files(axis):
    norm = sorted(
        EVO.glob(f"{axis}_norm_*.png"),
        key=lambda p: float(re.search(r"_norm_([0-9.]+)_", p.name).group(1)),
    )
    push = sorted(
        (p for p in EVO.glob(f"{axis}_push_*_win_*.png")),
        key=lambda p: float(re.search(r"_push_([0-9.]+)_win_", p.name).group(1)),
    )
    return norm, push


def panels_for(axis):
    if not EVO.exists():
        return [(None, "media archive not found")] * 4
    norm, push = axis_files(axis)
    panels = []
    if norm:
        indices = (
            [0, len(norm) // 2, len(norm) - 1]
            if len(norm) >= 3
            else list(range(len(norm)))
        )
        for index in indices:
            panels.append((norm[index], f"norm {strength_from_name(norm[index]):.2f}"))
    if push:
        panels.append((push[-1], f"push {strength_from_name(push[-1]):.2f}"))
    if len(panels) < 4 and push:
        for item in push:
            label = f"push {strength_from_name(item):.2f}"
            if (item, label) not in panels:
                panels.append((item, label))
    while len(panels) < 4:
        panels.append((None, "missing"))
    return panels[:4]


def build_atlas(axis_rows, survival_rows):
    records = []
    for row in axis_rows:
        if not row["label_push"] and row["dosed"] == "no":
            continue
        if row["axis"] not in AXIS_LABELS["axes"]:
            continue
        axis = row["axis"]
        thumb = OUT_THUMBS / f"{axis}.webp"
        if not thumb.exists() and (OUT_THUMBS / f"{axis}.png").exists():
            thumb = OUT_THUMBS / f"{axis}.png"
        records.append(
            {
                "axis": axis,
                "name": axis.replace("z_", "").replace("_", " "),
                "category": row["category"],
                "label": row["label_push"] or "unmeasured",
                "strict": row["strict_push"] or "",
                "cos": row["cos"],
                "clip_delta": row["clip_delta"],
                "edge_ratio": row["edge_ratio_top"],
                "lum_delta": row["lum_delta_max"],
                "onset": row["wash_onset"],
                "strength": row["top_strength"],
                "monotone": row["monotone"],
                "thumb": f"thumbs/{thumb.name}" if thumb.exists() else "",
            }
        )
    records.sort(
        key=lambda r: (
            CATEGORY_ORDER.index(r["category"])
            if r["category"] in CATEGORY_ORDER
            else 99,
            r["name"],
        )
    )
    (OUT_ATLAS / "atlas.json").write_text(json.dumps(records, indent=2))
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    holds = sum(1 for r in records if r["label"] == "holds")
    strict = sum(1 for r in records if r["strict"] == "holds")
    washes = sum(1 for r in records if r["label"] == "washes")
    unmeasured = sum(1 for r in records if r["label"] == "unmeasured")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Helm — Axis Atlas</title>
<style>
:root {{
  --bg:#0a0a0a; --panel:#111113; --line:#232327; --ink:#ededed;
  --muted:#9ca3af; --accent:#818cf8; --green:#22c55e; --red:#ef4444;
  --gray:#6b7280; --light:#f59e0b; --camera:#38bdf8; --material:#a78bfa;
  --atmosphere:#34d399; --geometry:#fb7185;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
header {{ padding:34px 28px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:clamp(24px,4vw,42px); letter-spacing:.16em; font-weight:600; }}
.sub {{ color:var(--muted); font-size:12px; line-height:1.7; max-width:820px; }}
.stats {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:18px; font-size:12px; }}
.stat b {{ display:block; font-size:20px; letter-spacing:.04em; }}
.stat span {{ color:var(--muted); }}
.controls {{ position:sticky; top:0; z-index:5; background:rgba(10,10,10,.94); backdrop-filter:blur(8px); padding:14px 28px; border-bottom:1px solid var(--line); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
input[type=search] {{ background:#151518; border:1px solid var(--line); color:var(--ink); padding:9px 12px; border-radius:7px; min-width:230px; font:inherit; }}
button, select {{ background:#151518; border:1px solid var(--line); color:var(--ink); padding:8px 11px; border-radius:7px; cursor:pointer; font:inherit; font-size:11px; }}
button.active {{ border-color:var(--accent); color:var(--accent); background:#1a1a2e; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:16px; padding:22px 28px 60px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.thumb {{ width:100%; aspect-ratio:18/5; object-fit:cover; background:#17171a; display:block; }}
.body {{ padding:12px 13px 14px; }}
.title {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
.title h2 {{ margin:0; font-size:13px; letter-spacing:.04em; }}
.badge {{ font-size:9px; padding:3px 6px; border-radius:99px; border:1px solid; white-space:nowrap; }}
.holds {{ color:var(--green); border-color:#14532d; background:#052e16; }}
.washes {{ color:var(--red); border-color:#7f1d1d; background:#2a0a0a; }}
.unmeasured {{ color:var(--gray); border-color:#374151; background:#111827; }}
.chips {{ display:flex; gap:6px; margin:8px 0 10px; flex-wrap:wrap; }}
.chip {{ font-size:9px; color:var(--muted); border:1px solid var(--line); padding:2px 6px; border-radius:99px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; font-size:10px; }}
.metrics span {{ color:var(--muted); display:block; font-size:8px; letter-spacing:.08em; }}
.metrics b {{ font-weight:500; }}
.light {{ color:var(--light); }} .camera {{ color:var(--camera); }}
.material {{ color:var(--material); }} .atmosphere {{ color:var(--atmosphere); }}
.geometry {{ color:var(--geometry); }}
footer {{ color:var(--muted); font-size:10px; padding:0 28px 40px; line-height:1.8; }}
a {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>H E L M — AXIS ATLAS</h1>
  <div class="sub">55 captured directions on Z-Image: 53 labelled, 2 unmeasured. Each card is the same prompt and seed at three <em>norm</em> strengths, plus the single <em>push</em> render available. The metrics use the top push dose. Green means the direction kept structure; red means it washed out. The numbers are in <a href="../data/axes.csv">axes.csv</a>; the method is in <a href="../README.md">the study README</a>.</div>
  <div class="stats">
    <div class="stat"><b>{len(records)}</b><span>axes</span></div>
    <div class="stat"><b>{holds}</b><span>holds</span></div>
    <div class="stat"><b>{washes}</b><span>washes</span></div>
    <div class="stat"><b>{strict}</b><span>strict holds</span></div>
    <div class="stat"><b>{unmeasured}</b><span>unmeasured</span></div>
  </div>
</header>
<div class="controls">
  <input id="q" type="search" placeholder="search axis…" autocomplete="off">
  <button data-cat="all" class="active">all</button>
  <button data-cat="light">light</button>
  <button data-cat="camera">camera</button>
  <button data-cat="material">material</button>
  <button data-cat="atmosphere">atmosphere</button>
  <button data-cat="geometry">geometry</button>
  <button data-status="holds">holds</button>
  <button data-status="washes">washes</button>
  <button data-status="unmeasured">unmeasured</button>
  <select id="sort">
    <option value="category">sort: family</option>
    <option value="clip">sort: ΔCLIP</option>
    <option value="edge">sort: edge retained</option>
    <option value="name">sort: name</option>
  </select>
</div>
<main class="grid" id="grid"></main>
<footer>
  Source of truth: <code>docs/data/axis-labels.json</code>, <code>docs/data/zimage-78-axes.json</code> and the 385-job provenance index. Atlas generated by <code>study/src/build_study.py</code>. Images are outside git; thumbnails are derived from the media archive. Code MIT; labels and metadata CC BY 4.0.
</footer>
<script>
const DATA = {payload};
const ORDER = ["light","camera","material","atmosphere","geometry"];
let cat = "all", status = "all", query = "", sort = "category";
const grid = document.getElementById("grid");
const q = document.getElementById("q");
const sortEl = document.getElementById("sort");
function filter() {{
  let rows = DATA.filter(r =>
    (cat === "all" || r.category === cat) &&
    (status === "all" || r.label === status) &&
    (!query || (r.name + " " + r.axis).includes(query))
  );
  if (sort === "name") rows.sort((a,b) => a.name.localeCompare(b.name));
  if (sort === "clip") rows.sort((a,b) => (parseFloat(b.clip_delta)||-9) - (parseFloat(a.clip_delta)||-9));
  if (sort === "edge") rows.sort((a,b) => (parseFloat(b.edge_ratio)||-9) - (parseFloat(a.edge_ratio)||-9));
  if (sort === "category") rows.sort((a,b) => ORDER.indexOf(a.category) - ORDER.indexOf(b.category) || a.name.localeCompare(b.name));
  grid.innerHTML = rows.map(card).join("");
}}
function card(r) {{
  const badge = r.label === "holds" ? "holds" : r.label === "washes" ? "washes" : "unmeasured";
  const strict = r.strict === "holds" ? "strict ✓" : (r.strict ? "strict: " + r.strict : "");
  const mono = r.monotone === "yes" ? "monotone" : (r.monotone === "no" ? "non-monotone" : "");
  const img = r.thumb
    ? `<img class="thumb" loading="lazy" src="${{r.thumb}}" alt="${{r.name}} at three strengths">`
    : `<div class="thumb" style="display:flex;align-items:center;justify-content:center;color:#4b5563;font-size:10px">no thumbnail</div>`;
  const delta = r.clip_delta ? (parseFloat(r.clip_delta) > 0 ? "+" : "") + r.clip_delta : "—";
  return `<article class="card">
    ${{img}}
    <div class="body">
      <div class="title"><h2>${{r.name}}</h2><span class="badge ${{badge}}">${{badge}}</span></div>
      <div class="chips"><span class="chip ${{r.category}}">${{r.category}}</span><span class="chip">cos ${{r.cos || "—"}}</span><span class="chip">top push ${{r.strength || "—"}}</span>${{strict ? `<span class="chip">${{strict}}</span>` : ""}}${{mono ? `<span class="chip">${{mono}}</span>` : ""}}</div>
      <div class="metrics">
        <div><span>ΔCLIP</span><b>${{delta}}</b></div>
        <div><span>EDGE ×</span><b>${{r.edge_ratio || "—"}}</b></div>
        <div><span>LUM Δ</span><b>${{r.lum_delta || "—"}}</b></div>
        <div><span>ONSET</span><b>${{r.onset || "—"}}</b></div>
      </div>
    </div>
  </article>`;
}}
document.querySelectorAll("[data-cat]").forEach(b => b.onclick = () => {{
  cat = b.dataset.cat;
  document.querySelectorAll("[data-cat]").forEach(x => x.classList.toggle("active", x === b));
  filter();
}});
document.querySelectorAll("[data-status]").forEach(b => b.onclick = () => {{
  status = status === b.dataset.status ? "all" : b.dataset.status;
  document.querySelectorAll("[data-status]").forEach(x => x.classList.toggle("active", x.dataset.status === status));
  filter();
}});
q.oninput = () => {{ query = q.value.toLowerCase(); filter(); }};
sortEl.onchange = () => {{ sort = sortEl.value; filter(); }};
filter();
</script>
</body>
</html>
"""
    (OUT_ATLAS / "index.html").write_text(html)


def content_rows(axis_rows):
    specials = [
        (
            "special",
            "Three hypotheses we killed",
            "negative-results",
            "NEGATIVE-RESULTS.md",
            "The predictor, the probe estimator, and norm/clamp.",
        ),
        (
            "special",
            "Helm vs prompt mixing",
            "prompt-mix",
            "charts/07_prompt_mix.png",
            "A step is not a slider.",
        ),
        (
            "special",
            "Which models are measured?",
            "model-matrix",
            "charts/09_model_matrix.png",
            "Two measured, five not.",
        ),
        (
            "special",
            "The axis atlas",
            "atlas",
            "atlas/index.html",
            "55 directions, one page.",
        ),
        (
            "special",
            "How this study is built",
            "method",
            "DATASET.md",
            "From renders to tables.",
        ),
    ]
    hooks = {
        "light": "Same prompt, same seed, one light direction.",
        "camera": "Same scene, one camera direction.",
        "material": "Surface response at three doses.",
        "atmosphere": "Atmosphere without rewriting the prompt.",
        "geometry": "Composition, not a new scene.",
    }
    rows = []
    for row in axis_rows:
        if row["label_push"] not in ("holds", "washes", "unmeasured"):
            continue
        if row["label_push"] == "unmeasured":
            continue
        name = row["axis"].replace("z_", "").replace("_", " ")
        angle = hooks.get(row["category"], "")
        if row["label_push"] == "holds":
            angle += " It holds; the caption carries the usable band."
        else:
            angle += " It washes; the caption names the failure mode."
        rows.append(
            {
                "kind": "axis",
                "week": "",
                "slot": "",
                "title": f"Axis — {name}",
                "axis": row["axis"],
                "category": row["category"],
                "label": row["label_push"],
                "asset": f"atlas/thumbs/{row['axis']}.webp",
                "angle": angle,
            }
        )
    rows.sort(
        key=lambda r: (
            CATEGORY_ORDER.index(r["category"])
            if r["category"] in CATEGORY_ORDER
            else 99,
            r["axis"],
        )
    )
    for index, row in enumerate(rows):
        row["week"] = str(index // 2 + 1)
        row["slot"] = "A" if index % 2 == 0 else "B"
    for offset, item in enumerate(specials):
        rows.append(
            {
                "kind": item[0],
                "week": str(offset + 1),
                "slot": "special",
                "title": item[1],
                "axis": "",
                "category": "",
                "label": "",
                "asset": item[3],
                "angle": item[4],
            }
        )
    return rows


def build_readme(labels, axes78, axis_rows, survival, bench_rows, models, prov_rows):
    labelled = sum(1 for r in axis_rows if r["label_push"] in ("holds", "washes"))
    holds = sum(1 for r in axis_rows if r["label_push"] == "holds")
    strict = sum(1 for r in axis_rows if r["strict_push"] == "holds")
    washes = labelled - holds
    unmeasured = sum(1 for r in axis_rows if r["label_push"] == "unmeasured")
    hold_rate = pct(holds, labelled)
    strict_rate = pct(strict, labelled)
    best = max(
        (r for r in survival if r["labelled"]),
        key=lambda r: (float(r["hold_rate"]), r["labelled"]),
    )
    worst = min(
        (r for r in survival if r["labelled"]),
        key=lambda r: (float(r["hold_rate"]), -r["labelled"]),
    )
    survivors = [r for r in axis_rows if r["label_push"] == "holds"]
    survivors.sort(key=lambda r: float(r["clip_delta"] or 0), reverse=True)
    survivor_rows = [
        [
            r["axis"].replace("z_", ""),
            r["category"],
            fmt(float(r["clip_delta"]), 3),
            fmt(float(r["edge_ratio_top"]), 2),
            fmt(float(r["lum_delta_max"]), 2),
            fmt(float(r["cos"]), 3),
            fmt(float(r["top_strength"]), 2),
            "yes" if r["strict_push"] == "holds" else "",
        ]
        for r in survivors
    ]
    category_rows = [
        [
            r["category"],
            r["labelled"],
            r["holds"],
            r["washes"],
            r["hold_rate"] + "%",
            r["strict_holds"],
            r["strict_rate"] + "%",
        ]
        for r in survival
    ]
    bench_order = [
        "full_s11",
        "full_s23",
        "helm0.5_s11",
        "helm0.5_s23",
        "helm1.0_s11",
        "mix0.9_s11",
        "mix0.8_s11",
        "mix0.7_s11",
        "mix0.5_s11",
        "mix0.3_s11",
    ]
    bench_by = {r["variant"]: r for r in bench_rows}
    bench_table = [
        [v, bench_by[v]["lum_ratio"], bench_by[v]["edge_ratio"], bench_by[v]["p_empty"]]
        for v in bench_order
        if v in bench_by
    ]
    model_table = [
        [
            r["model"],
            r["module"],
            r["blocks"],
            r["dim"],
            r["default_block"],
            r["band"],
            r["strength"],
            r["status"],
        ]
        for r in models
    ]
    stage_counts = Counter(r["stage"] for r in prov_rows)
    mode_counts = Counter(r["mode"] for r in prov_rows)
    stage_line = ", ".join(f"{k} {v}" for k, v in stage_counts.most_common())
    mode_line = ", ".join(f"{k} {v}" for k, v in mode_counts.most_common())
    label_rows_count = sum(
        1
        + len(a.get("push") or [])
        + len(a.get("norm") or [])
        + len(a.get("clamp") or [])
        for a in labels["axes"].values()
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"""# Helm Study 01 — 78 directions, 55 dosed

**One line:** of 78 captured directions on Z-Image, 55 were dosed and 53 labelled; **{holds} hold** under the pre-registered criterion and **{strict}** under a stricter quality ceiling. The rest wash out. This directory is the derived, reproducible layer behind that sentence.

> This is a study of the evidence already in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md), [`docs/data/`](../docs/data) and the 385-job provenance index. It is not a new claim. The JSON files are the source of truth; everything here is generated by [`src/build_study.py`](src/build_study.py).

## Headline

| stage | n |
|---|---:|
| captured | {len(axes78)} |
| dosed | {sum(1 for r in axes78 if r.get("dosed"))} |
| labelled | {labelled} |
| holds, pre-registered | **{holds}** ({hold_rate:.1f}%) |
| holds, strict ceiling | **{strict}** ({strict_rate:.1f}%) |
| washes | {washes} |
| unmeasured | {unmeasured} |

![survival funnel](charts/01_survival_funnel.png)

The funnel is the whole story: the interesting number is not how many directions we captured, it is how few survive a dose. `holds` means the CLIP contrast moved monotonically, edge density stayed at least 50% of baseline, and luminance stayed within 0.20 of baseline at every dose. `strict` adds a post-hoc ceiling (saturation <= 0.5, edge <= 2x baseline) and is reported next to the pre-registered label, not instead of it.

## Survival by family

{md_table(["family", "labelled", "holds", "washes", "hold rate", "strict holds", "strict rate"], category_rows)}

**Observation:** the best family here is **{best["category"]}** ({best["holds"]}/{best["labelled"]}, {best["hold_rate"]}%); the worst is **{worst["category"]}** ({worst["holds"]}/{worst["labelled"]}, {worst["hold_rate"]}%). Under the strict ceiling material falls to 1/10, so the family ranking is not stable across criteria. This is a description of this 55-axis set, not a law: the families are curated by hand and the n per family is small.

![survival by family](charts/02_holds_by_category.png)

## The survivors

{md_table(["axis", "family", "ΔCLIP", "edge ×", "lum Δ", "cos", "top push", "strict"], survivor_rows)}

![dose curves](charts/03_dose_curves.png)

**Observation:** several survivors are already at the edge of the ceiling at their top push dose. The `strict` column is empty for the three axes whose top push dose crosses the post-hoc saturation/edge limit (`saturated_vs_muted_stock`, `specular_reflection_density`, `specular_reflection_intensity`); those three are strict washes, not strict holds. The primary labels are unchanged; use the `safe_max` in the axis header, not the top push dose, as the production band. A hold is also not a promise of a large effect: `motion_blur` holds the criterion with a ΔCLIP of -0.054.

## The screen that failed

![cos vs survival](charts/04_cos_vs_survival.png)

**Observation:** cosine alignment with the mean activation does **not** separate holds from washes. The lowest survivor is at 0.310 and the highest wash is at 0.982; every hold sits inside the wash range. At n=23 the same metric looked like a screen; at n=78 it is a small-sample artifact. The numbers stay, the story is in [`NEGATIVE-RESULTS.md`](NEGATIVE-RESULTS.md).

## Where the wash happens

![wash map](charts/05_wash_map.png)

**Observation:** the wash is not evenly distributed. Some axes lose edge early; others keep structure to the top push dose and fail on saturation or luminance instead. A wash can also be a monotonicity failure with no edge collapse at all (`mixed_color_sources`, `symmetry`, `sun_angle`, `subject_scale`, `pro_mist_diffusion`); those have an empty `onset`. The `onset` column in [`data/axes.csv`](data/axes.csv) is the first strength that crosses any of the three structural failure conditions, so it is not a complete label on its own.

## Effect size

![effect size](charts/06_effect_size.png)

**Observation:** the axes that hold are not simply the ones with the largest CLIP move. A useful direction keeps edge ratio above 0.5 and still moves the contrast; a large ΔCLIP with a collapsed edge is the wash, not a stronger edit.

## Helm is not a prompt mixer

![prompt mix](charts/07_prompt_mix.png)

{md_table(["variant", "lum ×", "edge ×", "P(empty)"], bench_table)}

**Observation:** on this benchmark Helm at 0.5 keeps the scene (P(packed) = 1.0) and grades luminance and detail; the prompt mix is a no-op from 0.9 to 0.7 and flips the scene at 0.3. The fair design starts both methods from the same full-room prompt. One model, one contrast, two seeds; a LoRA baseline is still not measured.

## Model coverage

{md_table(["model", "module", "blocks", "dim", "default block", "band", "strength", "status"], model_table)}

![model matrix](charts/09_model_matrix.png)

**Observation:** only H3 and Z-Image have a measured band. LTX is wired but untested; Krea 2 has templates but no calibration entry; Flux is declared but not dosed; SD3 and Qwen-Image are refused until measured. No Helm claim rests on the unmeasured models.

## Provenance

The `385` jobs behind the 2026-10-19 window are indexed with `sha256`, graph hash and sidecar path: [`data/provenance.csv`](data/provenance.csv). By stage: {stage_line}. By mode: {mode_line}. The summary is [`data/provenance_summary.csv`](data/provenance_summary.csv); the original is [`docs/data/evolution-20261019/renders.jsonl`](../docs/data/evolution-20261019/renders.jsonl).

![provenance](charts/08_provenance.png)

## The atlas

[`atlas/index.html`](atlas/index.html) is the same 55 axes as a visual sheet: three norm strengths plus the single push render available (middle strength), filterable by family and outcome. The card metrics use the top push dose; the fourth panel is the middle-strength push render. It works from `file://`; the thumbnails are derived from the media archive and are not required to rebuild the tables.

## Reproduce

```bash
make -C study
# or
uv run --no-project --with matplotlib --with pillow --with numpy \\
  python study/src/build_study.py
```

The CSVs and charts need only the JSON in `docs/data/`. The atlas thumbnails additionally need the external media archive at `~/HAL-archive/helm-videos/build/v38-intake/z19/2026-10-19/evo/`; if it is missing the build still writes every table and chart and the atlas falls back to placeholders.

## Data and licence

- [`data/axes.csv`](data/axes.csv) — one row per captured direction (78).
- [`data/labels_long.csv`](data/labels_long.csv) — tidy rows for baseline/push/norm/clamp ({label_rows_count}).
- [`data/doses.csv`](data/doses.csv) — the 163 measured dose points.
- [`data/window.csv`](data/window.csv) — the 55 step-window points.
- [`data/bench_prompt_mix.csv`](data/bench_prompt_mix.csv) — the fair prompt-mix benchmark.
- [`data/provenance.csv`](data/provenance.csv) — the 385 jobs.

Code MIT; labels and metadata CC BY 4.0. The renders themselves live outside git; see [`DATASET.md`](DATASET.md) for the data card and exclusions.

## Limitations

- One model (Z-Image), one contrast pair, seed 11, three strengths per axis; the H3 dose-response is separate and smaller.
- CLIP is a useful low-strength probe and a confounded high-strength one (it tracks the tonal shift, not only the content).
- The families are curated, not measured; the per-family rates are descriptive.
- `strict_push` is post-hoc; the pre-registered labels are `labels.push`.
- The mask localises the injection, not the visible effect: attention spreads the change outside the mask.
- The atlas thumbnails are 260 px crops for browsing, not evidence.

---

Built {stamp} from `docs/data/axis-labels.json`, `docs/data/zimage-78-axes.json`, `docs/data/zimage-dose-response.json`, `docs/data/bench2-vs-prompt-mix.json`, `docs/data/evolution-20261019/renders.jsonl` and `calibration.json`.
"""
    (STUDY / "README.md").write_text(text)


def build_summary(labels, axes78, axis_rows, survival, bench_rows, prov_rows):
    labelled = sum(1 for r in axis_rows if r["label_push"] in ("holds", "washes"))
    holds = sum(1 for r in axis_rows if r["label_push"] == "holds")
    strict = sum(1 for r in axis_rows if r["strict_push"] == "holds")
    best = max(
        (r for r in survival if r["labelled"]), key=lambda r: float(r["hold_rate"])
    )
    survivors = [r for r in axis_rows if r["label_push"] == "holds"]
    lowest = min(survivors, key=lambda r: float(r["cos"])) if survivors else None
    washes = [r for r in axis_rows if r["label_push"] == "washes"]
    highest = max(washes, key=lambda r: float(r["cos"])) if washes else None
    lines = [
        "# Helm study — summary",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## Headline",
        "",
        f"- Captured: **{len(axes78)}** directions.",
        f"- Dosed: **{sum(1 for r in axes78 if r.get('dosed'))}**.",
        f"- Labelled: **{labelled}**.",
        f"- Holds (pre-registered): **{holds}/{labelled}** ({pct(holds, labelled):.1f}%).",
        f"- Holds (strict ceiling): **{strict}/{labelled}** ({pct(strict, labelled):.1f}%).",
        "",
        "## What the data says",
        "",
        f"- Best family by hold rate: **{best['category']}** ({best['holds']}/{best['labelled']}, {best['hold_rate']}%).",
    ]
    if lowest and highest:
        lines.append(
            f"- The mean-activation screen does not separate: lowest-cos survivor **{lowest['axis'].replace('z_', '')}** at {lowest['cos']}, highest-cos wash **{highest['axis'].replace('z_', '')}** at {highest['cos']}."
        )
    lines += [
        f"- Push beats norm and clamp under the same criterion: **{holds}** vs **1** vs **2** holds.",
        "- The prompt mix is a step; Helm is a graded control at the same framing.",
        "- Only H3 and Z-Image have a measured band; five models are wired but unmeasured.",
        "",
        "## Files",
        "",
        "- `data/` — tidy CSVs generated from the JSON.",
        "- `results/` — metrics, survival by family, model matrix, content calendar.",
        "- `charts/` — the nine figures used in the README.",
        "- `atlas/` — the HTML axis atlas.",
        "",
        "The source JSON, provenance hashes and limitations are in `docs/` and `study/DATASET.md`.",
        "",
    ]
    (OUT_RESULTS / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-thumbs", action="store_true")
    args = parser.parse_args()

    global AXIS_LABELS
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    OUT_ATLAS.mkdir(parents=True, exist_ok=True)
    OUT_THUMBS.mkdir(parents=True, exist_ok=True)

    labels = read_json(DATA / "axis-labels.json")
    axes78 = read_json(DATA / "zimage-78-axes.json")
    doses = read_json(DATA / "zimage-dose-response.json")
    bench = read_json(DATA / "bench2-vs-prompt-mix.json")
    calibration = read_json(ROOT / "calibration.json")
    prov_lines = (
        (DATA / "evolution-20261019" / "renders.jsonl").read_text().splitlines()
    )
    AXIS_LABELS = labels

    missing = sorted(set(CATEGORY) - {r["axis"] for r in axes78})
    if missing:
        raise SystemExit(f"uncategorised axes: {missing}")

    axis_rows = build_axis_rows(labels, axes78)
    long_rows = build_long_rows(labels)
    window_rows = build_window_rows(labels)
    dose_rows = build_dose_rows(doses)
    bench_rows = build_bench_rows(bench)
    prov_rows = build_provenance_rows(prov_lines)
    survival = survival_rows(axis_rows)
    models = model_rows(calibration)

    write_csv(
        OUT_DATA / "axes.csv",
        list(axis_rows[0].keys()),
        axis_rows,
    )
    write_csv(
        OUT_DATA / "labels_long.csv",
        ["axis", "category", "mode", "strength", "lum", "sat", "edge", "clip", "label"],
        long_rows,
    )
    write_csv(
        OUT_DATA / "window.csv",
        ["axis", "category", "strength", "lum", "sat", "edge", "clip", "edge_ratio"],
        window_rows,
    )
    write_csv(
        OUT_DATA / "doses.csv",
        [
            "axis",
            "category",
            "s_ref",
            "strength",
            "lum",
            "sat",
            "edge",
            "dlum",
            "dsat",
            "dedge",
        ],
        dose_rows,
    )
    write_csv(
        OUT_DATA / "bench_prompt_mix.csv",
        [
            "variant",
            "family",
            "level",
            "seed",
            "lum",
            "sat",
            "edge",
            "p_empty",
            "p_packed",
            "lum_ratio",
            "edge_ratio",
        ],
        bench_rows,
    )
    write_csv(
        OUT_DATA / "provenance.csv",
        [
            "axis",
            "stage",
            "mode",
            "strength",
            "status",
            "output",
            "sha256",
            "graph_sha256",
            "job_id",
        ],
        prov_rows,
    )
    write_csv(
        OUT_RESULTS / "metrics.csv",
        list(axis_rows[0].keys()),
        axis_rows,
    )
    write_csv(
        OUT_RESULTS / "survival_by_category.csv",
        [
            "category",
            "labelled",
            "holds",
            "washes",
            "hold_rate",
            "strict_holds",
            "strict_rate",
            "median_clip_delta",
            "median_edge_ratio",
        ],
        survival,
    )
    write_csv(
        OUT_RESULTS / "model_matrix.csv",
        [
            "model",
            "module",
            "blocks",
            "dim",
            "default_block",
            "band",
            "strength",
            "status",
        ],
        models,
    )
    prov_summary = [
        {"stage": stage, "mode": mode, "jobs": count}
        for (stage, mode), count in sorted(
            Counter((r["stage"], r["mode"]) for r in prov_rows).items()
        )
    ]
    write_csv(
        OUT_DATA / "provenance_summary.csv",
        ["stage", "mode", "jobs"],
        prov_summary,
    )
    calendar = content_rows(axis_rows)
    write_csv(
        OUT_RESULTS / "content_calendar.csv",
        [
            "kind",
            "week",
            "slot",
            "title",
            "axis",
            "category",
            "label",
            "asset",
            "angle",
        ],
        calendar,
    )

    setup_matplotlib()
    chart_funnel(labels, axes78)
    chart_categories(survival)
    chart_dose_curves(labels)
    chart_cos(labels, axes78)
    chart_wash_map(axis_rows)
    chart_effect(axis_rows)
    chart_bench(bench_rows)
    chart_provenance(prov_rows)
    chart_model_matrix(models)

    if not args.skip_thumbs:
        if EVO.exists():
            for row in axis_rows:
                if row["axis"] in AXIS_LABELS["axes"]:
                    make_thumb(row["axis"], panels_for(row["axis"]))
        else:
            print(f"media archive not found: {EVO} (atlas thumbnails skipped)")

    build_atlas(axis_rows, survival)
    build_readme(labels, axes78, axis_rows, survival, bench_rows, models, prov_rows)
    build_summary(labels, axes78, axis_rows, survival, bench_rows, prov_rows)

    print(
        f"axes: {len(axis_rows)}  labelled: {sum(1 for r in axis_rows if r['label_push'] in ('holds', 'washes'))}"
    )
    print(
        f"holds: {sum(1 for r in axis_rows if r['label_push'] == 'holds')}  strict: {sum(1 for r in axis_rows if r['strict_push'] == 'holds')}"
    )
    print(
        f"charts: {len(list(OUT_CHARTS.glob('*.png')))}  thumbs: {len(list(OUT_THUMBS.glob('*')))}"
    )


AXIS_LABELS = {}


if __name__ == "__main__":
    main()
