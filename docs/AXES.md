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
