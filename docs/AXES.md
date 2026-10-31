# Shipped axes: provenance

The seven Z-Image axes below were captured on 2026-10-18 (block 15, dim 3840, model `NextDiT`) and are the axes that hold under the pre-registered criterion in [EXPERIMENTS.md](EXPERIMENTS.md) section F (`strict_push=holds`, monotone) in [`study/data/axes.csv`](../study/data/axes.csv).

The tensor bytes are exactly as captured. Only the safetensors header was rewritten, to record `safe_max` (the highest strength tested and held, from `axes.csv` `top_strength`) and `safe_max_source: benchmark (axes.csv)`. `tensor sha256` is the hash of the 15,360 tensor bytes, so it proves the vector was not altered; `file sha256` is the hash of the shipped file.

| axis | safe_max | file sha256 | tensor sha256 |
|:--|--:|:--|:--|
| `z_raking_light_angle` | 2.8064 | `97f21d8e7e2477a286dd24f8eddb7dcc5fa0c0906205f9cef885c69f9d22a827` | `9126f7a7bff2e0018bc03698e4fb810a980247a6afd4bc4d72df1f2582144c70` |
| `z_volumetric_beams` | 2.0101 | `e58e90d16a07ed98892e4548f25e9f6935bc818a13e9fd5a7d1102dfe4ed1ffe` | `38437f91109c6c04913a62807e200273c8339b3d44cd94689b625c46f5d3fdc3` |
| `z_overcast_vs_direct` | 2.3931 | `a8e444b8e2682cc502a7cd80cb79e687d8f8568a0acef01355db7580a340bfad` | `96617a4c9a85cc06aebddb4e02b5695eed68efddef63c19c76a2cd1eaa89e8d3` |
| `z_surface_reflection_intensity` | 3.036 | `51d2f80e526ff74bb0117ee7908fb9e7e7762b43d907dabab8ef23caf26affe4` | `6db9d03c2a53860d2049c7e604d0498731e6c1aca6113916e6d84633bf12dcd0` |
| `z_motion_blur` | 0.4913 | `d003f7fadd22c2a64e2d4c9a554ba38c20f2c936f9191d180b3e213e4b8a324a` | `d1edb4e7c1129519ae8c6848b94d5dcc4943f03e2b7278fa244c6f6a021604fd` |
| `z_aisle_width` | 4.5327 | `e232d4591e0e7323233c9c4d4670f80eb08ed4b8f46c2112bc690644d28e379c` | `0dac334f77d8864d6c4bf67a3f4f793fcc1f352b5f0500b66be4d07d76100482` |
| `z_corridor_orientation` | 9.5532 | `627d0f437bb136a2882e85e5fd1de8bbf1cefd34c93923ceeb51722092737308` | `c70df057ff67a1175c0252d07c6315f231dc195ea0cd2ea67e97d2b9e0231b6e` |

`z_room_density`, `density` and `mood` were shipped earlier; their evidence is in EXPERIMENTS.md sections A, B and E.

## Preview renders

Before/after thumbnails in `web/previews/` and the template previews are real Helm renders. Same prompt and seed (11) in each pair; Amount 0 against a strength at or below the axis' `safe_max`; nothing from the washed range. Axes with no clear real pair (the three Subtle axes below, and the "less clutter" Z-Image axis, whose only stored pair goes past its safe range) have no thumbnail.

| Preview | Source run | Render files | After strength | Amount |
|:--|:--|:--|--:|--:|
| `z_raking_light_angle.webp` | Z-Image 78-axis dose run, seed 11 | `z_raking_light_angle_s0` / `_s3.0` | 2.8064 | 1.0 |
| `z_overcast_vs_direct.webp` | same | `z_overcast_vs_direct_s0` / `_s3.0` | 2.3931 | 1.0 |
| `z_aisle_width.webp` | same | `z_aisle_width_s0` / `_s3.0` | 4.5327 | 1.0 |
| `z_corridor_orientation.webp` | same | `z_corridor_orientation_s0` / `_s3.0` | 9.5532 | 1.0 |
| `density.webp` | MiniMax H3 density dose sweep (`docs/images/density_sweep.png`, EXPERIMENTS.md section A) | mid frames at strength 0 and 0.20 | 0.20 | 0.67 |
| `Helm 1. Start here (Z-Image).jpg` | same frames as `z_raking_light_angle.webp` | `z_raking_light_angle_s0` / `_s3.0` | 2.8064 | 1.0 |
| `Helm 2. Video (MiniMax H3).jpg` | same frames as `density.webp` | mid frames at strength 0 and 0.20 | 0.20 | 0.67 |

The images show only plain labels ("Before", "After (Amount 1.0)"); the strengths in the table above are the technical numbers. The Z-Image renders are 704 px, resized and stripped of metadata; `tools/render_previews.py` re-labels the frames and builds the square template thumbnails (no GPU). `Helm 3. Make your own axis.jpg` is a layout drawing, not a render, because that axis is made on the spot. The Z-Image base prompt is not recorded alongside the render files, so it is not quoted here.

## Essentials and Subtle

The Dial lists the shipped axes in two sections, in the order of `axes_catalog.json`.

Essentials are ordered by **visible impact**, a score computed from [`study/data/axes.csv`](../study/data/axes.csv): the saturation shift at the top dose (`sat_top`) plus the peak luminance shift (`lum_delta_max`) plus `|edge_ratio_top - 1|`. It is stored as `visible_impact` in the catalog and checked by `tests/test_dial.py`. The two "less clutter" axes come from sweeps in [EXPERIMENTS.md](EXPERIMENTS.md) sections A and E rather than `axes.csv`, so they have no score and sit after the scored Essentials.

**Subtle** holds the three axes that hold in the benchmark (`strict_push=holds`, monotone) but show no visible effect in our renders. Their scores are below every Essential.

| Dial section | axis | visible_impact | safe_max |
|:--|:--|--:|--:|
| Essentials | `z_corridor_orientation` | 1.589 | 9.5532 |
| Essentials | `z_aisle_width` | 1.334 | 4.5327 |
| Essentials | `z_overcast_vs_direct` | 0.966 | 2.3931 |
| Essentials | `z_raking_light_angle` | 0.661 | 2.8064 |
| Essentials | `z_room_density` (less clutter, Z-Image) | n/a | 0.25 |
| Essentials | `density` (less clutter, MiniMax H3) | n/a | 0.3 |
| Subtle | `z_surface_reflection_intensity` | 0.348 | 3.036 |
| Subtle | `z_volumetric_beams` | 0.204 | 2.0101 |
| Subtle | `z_motion_blur` | 0.108 | 0.4913 |
