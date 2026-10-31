# UX notes: how Helm presents itself, and why

Helm's promise is **one slider moves one idea, no prompt rewrite**. A first-time user should know in ten
seconds where the Dial is and what to press. These are the conventions we adopted after reading how the
official ComfyUI templates and a handful of mature node packs do it, and what we checked in a real ComfyUI
(0.43.2 code with frontend 1.57.4, CPU, no model weights).

## What we found (facts, with sources)

**How ComfyUI lists a custom node's templates is much poorer than the official index.**

- The server (`app/custom_node_manager.py`) answers `/api/workflow_templates` with only a map of
  `custom node folder -> template file names`. It scans `example_workflows/*.json` (also `example`,
  `examples`, `workflow`, `workflows`).
- The frontend then builds each card from the **file name alone**: title = file name, description = the file
  name with `-` and `_` replaced by spaces, thumbnail = `<file name>.jpg` next to the workflow (the
  `mediaSubtype` is hard-coded to `jpg`). There is no index for custom nodes, in the installed frontend or in
  the current `ComfyUI_frontend` main.
- The rich `templates/index.json` (title, description, tags, models, `thumbnailVariant`, `minComfyUIVersion`) belongs
  to the official `Comfy-Org/workflow_templates` repo and is not read for custom nodes
  ([`docs/SPEC.md`](https://github.com/Comfy-Org/workflow_templates/blob/main/docs/SPEC.md) there).
- The sort order is not under our control (it came out 2, 3, 1 on a search). Names start with `Helm 1.`,
  `Helm 2.`, `Helm 3.` so that the browser's "Alphabetical (A-Z)" sort gives the right order.
- A template always opens **fit to the window**: `extra.ds` is ignored for templates (frontend
  `loadGraphData`, template branch calls `fitView`). Below about 60% zoom the canvas draws nodes as empty
  boxes (only group titles and custom-drawn widgets stay visible). So the graph has to be compact, not just tidy.
- Frontend-native `MarkdownNote` is available (it has no Python definition, so it is not in `/object_info`).
- A node's `DESCRIPTION` shows in the node search preview, and `SEARCH_ALIASES` feeds the search.
- Node help (the "i" panel) is read from `<pack web dir>/docs/<NodeClassName>.md`
  (frontend `NodeHelpService`, `/extensions/<pack>/docs/`).

**What the official templates do** (`Comfy-Org/workflow_templates`, `templates/`):

- `index.json`: titles are `Model: Task` ("Z-Image-Turbo: Text to Image", "MiniMax H3: Image to Video"), tags are
  task words, `models` lists the model family, `thumbnailVariant: "compareSlider"` is used for every before/after
  template (`utility_seedvr2_image_upscale`, relight templates) with two thumbnails `-1` and `-2`.
- Thumbnails: `<name>-1.webp`, square, 350 to 400 px, under 1 MB (`docs/SPEC.md`: "512x512 or 768x768", "WebP").
- Workflows are almost all **one subgraph node plus a few top-level nodes**: `image_z_image_turbo` has 3 top-level
  nodes (a note, the subgraph, `SaveImage`); `video_minimax_h3_i2v` has 9. What you edit (`LoadImage`, `SaveVideo`)
  stays outside the subgraph.
- Notes are `MarkdownNote` (not `Note`), titled `Note: <topic>` ("Note: Model Links", "Note: Size Settings
  Reference"), colour `#222` / `#000`, a `##` heading, bullets, links to the model repo and the issue tracker.
- Groups are rare (`video_minimax_h3_i2v` has one, `#3f789e`, the default blue).

**What the packs do:**

| Pack | Naming and presentation | Notes and colours | Single "main" node |
|:--|:--|:--|:--|
| Kijai `ComfyUI-WanVideoWrapper` (`example_workflows/`, e.g. `wanvideo_2_1_14B_FLF2V_720P_example_02.json`) | Descriptive file names. Nodes retitled where the role matters ("Load Image: Start image", "Set_start_image"). | Many **short** `Note` nodes (`#432` / `#653`) placed right next to the node they explain; one `MarkdownNote` for model links; some nodes tinted (`#2a363b` slate, `#332922` brown); one blue group for an optional branch ("ComfyUI text encoding alternative"). | Sampler is the hub; loaders are separate on purpose. |
| `kijai/ComfyUI-KJNodes` (236 nodes) | Categories `KJNodes/<topic>`. Tooltips on 230 of 1363 inputs in a `/object_info` dump (17%). | Only one example workflow. | A grab bag; no entry point. A negative example for onboarding. |
| `ComfyUI-VideoHelperSuite` | Display names carry an emoji suffix; node descriptions are long HTML. Tooltips on 0 of 138 inputs in a `/object_info` dump. | Uses its own preview widget instead of notes. | `Video Combine` is the one node everybody finds. |
| `rgthree-comfy` | One word per node (`Seed`, `Reroute`, `Context`) with a ` (rgthree)` suffix; README leads with a screenshot per node and a collapsed "See more". Tooltips on 2 of 103 inputs. | Right-click "Node Help" for in-app help. | `Seed` and `Context` are the entry points. |
| `numz/ComfyUI-SeedVR2_VideoUpscaler` | Category `SEEDVR2`; main node is named after the job (`SeedVR2 Video Upscaler`), helpers say `(Down)Load DiT Model`. Tooltips on 41 of 41 inputs and a one-sentence description per node in a `/object_info` dump. | Examples are `.json` + `.jpg` pairs (`SeedVR2_simple_image_upscale.json/.jpg`), 9 to 11 nodes, two small `Note`s beside the two switches that need explaining. No groups. | One upscaler node plus two loader nodes: simple by default, advanced one hop away. |
| `ltdrdata/ComfyUI-Impact-Pack` | Examples are numbered and named by the job (`1-FaceDetailer`, `2-MaskDetailer`...), each with a `.jpg`. | Two role tints in `1-FaceDetailer.json`: blue (`#223` / `#335`) on the detector, SAM, FaceDetailer and preview nodes, red (`#322` / `#533`) on the prompts, latent, KSampler and decode; previews retitled ("Cropped (refined)", "Mask", "Enhanced"). | `FaceDetailer` is the "main" node; the rest are advanced. |
| `ComfyUI_MiniMaxH3_Director` | One `MiniMaxH3Director` node, category `MiniMaxH3`; advanced variants (`...Refine`, `...Conditioning`, `Director Groups`) sit in sub-categories. 103 of 127 inputs in the pack have tooltips (81%); the descriptions say what to connect where. | n/a | One node does the job; the rest are building blocks. |
| `NikoDemon80/ComfyUI-H3-Motion-Context` (`H3-Motion-Context-v0.7.json`) | Single example, 7 groups with plain titles ("Model Loading", "Prompting", "Sampling"), 4 subgraphs. | 4 `MarkdownNote`s titled `... - Quick Help` and `Dependencies & Models`, a "Length in Frames" note next to the input it explains, a note at the reference-image slot telling you which file to drag in. Inputs tinted differently from loaders. | A "Chain" node with Run / Approve buttons drives the workflow. |

## Conventions we adopted

1. **File name = title.** `Helm 1. Start here (Z-Image).json`, `Helm 2. Video (MiniMax H3).json`,
   `Helm 3. Make your own axis.json`. Because ComfyUI shows the file name, we write the title we want people to
   read (official style: model in brackets, number first so it sorts). No underscores.
2. **A square `.jpg` thumbnail with the same name** (512 px, no metadata). The card is square and the lookup is
   hard-coded to `.jpg`; PNG previews, which we shipped first, were never shown. Templates 1 and 2 use a
   left/right Before | After built from the real renders; template 3 uses a small diagram because its axis is made on
   the spot. Plain labels only: "Before" and "After (Amount 1.0)".
3. **`MarkdownNote` only, titled `Note: <topic>`, official colours (`#222` / `#000`).** Short, a heading, then bullets;
   the **three steps are the first thing you see** (top-left, the only node at the origin) and are written as
   headings so they read at the zoom the template opens at. A note beside the thing it explains
   (Kijai, Motion Context), not one wall of text.
4. **Colour means one thing.** Green = you edit it (the prompt). Orange = the Helm control (`Helm · Dial`,
   `Helm · Make Axis`). Grey = leave as it is (models). Blue = the result. The same colours are used for the
   group, the node inside it and the node class itself (`web/helm_axes.js` colours Helm nodes when you add one,
   so a node from search matches the templates). Lab nodes are slate, so they read as "advanced". Impact Pack
   and Kijai tint nodes by role the same way.
5. **Numbered group titles for the steps, large font.** `1 · Write your prompt`, `2 · Move the Dial`,
   `3 · Result: Before | After`. The non-step group is plain: `Models (leave as they are)`.
6. **Compact and collapsed.** Because templates always open fit-to-window, the graph is about 2000 x 1000
   units and the plumbing (loaders, sampling shift, KSamplers, decoders, stitch, save) is **collapsed to one-line
   pills** that keep their real titles. This is our substitute for the official subgraph: the Dial, the prompt and
   the picture stay on screen, nothing is hidden in a subgraph (a subgraph would hide the Dial's slider
   behind a promoted widget), and a user can still expand any pill. The Dial is also the widest node (500+ units).
7. **One main node, advanced ones one hop away** (SeedVR2, Impact Pack, MiniMax Director): `Helm · Dial` and
   `Helm · Make Axis` live in `Helm`; the five manual nodes live in `Helm/Lab` and say "Helm Lab ·" and, in their
   description, "Lab:". Node search shows `Helm · Dial` first.
8. **Tooltips on every input and a `DESCRIPTION` on every node**, plain words, no `safe_max`, `block` or `lambda`
   in anything a first-time user reads (SeedVR2, 41 of 41 inputs, and the MiniMax Director, 81%, are the models here; KJNodes 17%, rgthree 2%
   and VHS 0% are the counter-examples; figures from a `/object_info` dump of an install with these packs). Outputs have `OUTPUT_TOOLTIPS` on the two main nodes. Search aliases ("slider", "one idea",
   "light") on the main nodes so people find them by what they want.
9. **Say what the Dial is doing, in words.** A live caption under the slider ("Light angle (raking) at 60%",
   "(Dial off)" at 0) and the same sentence in the console. It is drawn by the node, so it is readable even
   when the canvas is zoomed out.
10. **Node help** for every main node in `web/docs/` (`HelmDial.md`, `HelmMakeAxis.md`, `HelmAxis.md`, `HelmShape.md`, `HelmMix.md`, `HelmApplyNode.md`; the "i" panel).
11. **Menu order by visible impact, with a Subtle section.** *Essentials* are sorted by how visible the change is
    (top-dose saturation shift + peak luminance shift + |edge ratio - 1|, from `study/data/axes.csv`; the numbers are
    in `axes_catalog.json` as `visible_impact`). Three axes that hold in the benchmark but show no visible effect
    in our renders (volumetric beams, surface reflections, motion blur) are listed under *Subtle*. A new Dial
    starts on *Light angle (raking)*, a plain, easy to read idea, not on the highest-impact one.
12. **Technical numbers live in the docs.** The preview images say "Before" and "After (Amount 1.0)". `safe_max` and
    strengths are in `docs/AXES.md`.

## Helm 0.5: the chain (what we added, and how it was checked)

Checked in a real ComfyUI 0.43.2 with frontend 1.57.4 (CPU, no weights, throwaway base directory).

13. **"What comes next" when a link is released on empty canvas.** What the frontend does (read from the installed
    bundle, then confirmed in the browser):
    - *Plain release.* New installs default to the **node search** (`Comfy.LinkRelease.Action` = search box, from
      install version 1.24.1; Shift = context menu). The search is opened with the slot type as a filter: every node that takes
      that type is listed first, then the nodes that take any type. For `HELM` that is exactly the four Helm nodes, so the
      next Helm node is already on screen. The order inside the list is fixed by the frontend (shortest node name first
      for an empty query, so *Mix, Axis, Shape, Apply*) and a pack cannot change it.
    - *Menu.* The link-release menu (Shift + release, or the plain release if the user chose it) lists
      `LiteGraph.slot_types_default_out[type]` for a dragged output and `slot_types_default_in[type]` for a dragged input.
      The stock tables are built from node registration order and cut to 5, so a plain `HELM` table has no idea what
      comes after *this* node and shows class names (`HelmApplyNode`).
    - *What `web/helm_flow.js` does.* It wraps `LGraphCanvas.showConnectionMenu`. Only when the link comes from (or goes to) a
      Helm node, it sets the table for that slot type to a short, ordered list for the moment the menu is open (LiteGraph
      checks the table again when an entry is clicked, so it is restored when the menu closes), and shows each node by its
      title. Every other node, and every other pack, sees the stock tables. If the wrap cannot be installed, nothing
      changes.
    - *The suggestions.* HELM out of Axis: Shape, Mix, Apply, Axis. Out of Shape: Apply, Mix. Out of Mix: Apply. MODEL out
      of Apply (and the Dial): KSampler, Basic Guider, CFG Guider, KSampler (Advanced). Dragging from a HELM input:
      Axis. The node that is created is connected to the first free slot of that type (Mix gets `helm_1`; Axis is
      chained through its optional `helm` input).
14. **Role colours.** Purple = Axis (WHAT), orange = Shape (HOW MUCH) and the one-node Dial, blue = Mix (COMBINE), grey =
    Apply (WHERE), slate = Lab. The same hex values are in `web/helm_axes.js` and in the templates, and
    `tests/test_templates.py` compares them. Groups: green = you edit, purple = the Axis group, orange = Helm controls,
    grey = leave alone, blue = the result.
15. **Progressive disclosure with `widget.hidden`.** Axis in library mode shows `source` and `idea` only; in prompts
    mode it swaps the idea menu for pairs, name, debias, steps, seed, cfg and block. Shape shows `split` for early/late,
    and `end_amount`/`easing`/`keyframes` for the ramp and keyframes curves. Hidden widgets keep their values and their place in
    `widgets_values`, so saved workflows are unchanged. A saved workflow keeps its own size; only a change of mode resizes.
16. **Captions in words** drawn by the node (readable when zoomed out): Axis (idea, or "New idea from 3 prompt pairs, saved as
    my_axis"), Shape ("-40% (less), late steps", "+50% -> 0% (ramp), all steps"), Mix (before a run: "Mixing 3 inputs, made
    independent"; after a run: the first line of its report, which Mix also sends to the UI as a `text` message).
17. **Templates 4 to 6 teach the chain by level**, with the same rules as 1 to 3: 4 is Axis -> Apply (two nodes), 5 is
    Axis x 3 -> Mix -> Shape -> Apply (with a note that Shape and Mix are experimental), 6 builds an Axis from three
    prompt pairs. Thumbnails are clean diagrams in the role colours (no frames are borrowed, because the default amount
    of the chain was never rendered as a Before | After pair). The drawing starts below the "Node graph" badge the card
    paints on the top-left. File names contain an arrow and a times sign; the template browser, the thumbnail lookup and
    the sort order all work with them.

## What we did not copy, and why

- **`index.json` / tags / `thumbnailVariant: compareSlider`.** Not read for custom nodes. If ComfyUI adds an index
  for custom-node templates, it will be one small file (`name`, `title`, `description`, `mediaType`, `mediaSubtype`,
  `tags`, `models`); until then the file name is the title.
- **Subgraphs.** See convention 6. Revisit if a future frontend lets a subgraph node show promoted widgets
  on the outside without being a black box.
- **Many small `Note` nodes next to every widget** (Kijai). Our users are not tuning samplers; the three
  steps and one note per group are enough.
- **Emoji in node names** (VHS). They make search results look busy and do not survive every font.

## Checking it

`tools/build_templates.py` regenerates the templates and `tools/render_previews.py` the thumbnails and labelled
previews (no GPU). `tests/test_templates.py` checks names, thumbnails, note style, colours, overlaps, compactness
and collapsed plumbing; `tests/test_dial.py` the menu order, tooltips and the plain-words line. The real-browser
pass (template browser, each template, the Dial menu, node search) was done against a throwaway ComfyUI base
directory with the pack symlinked in; no model weights, so only missing-model warnings appear.
