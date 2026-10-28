# Content calendar — from the atlas to the feed

The study already contains the content. There are **53 labelled axes** and **5 special posts**; at two posts per week that is **27 weeks** of axis posts without a new render. The full list, with the asset path and the angle, is [`results/content_calendar.csv`](results/content_calendar.csv).

## The rule

Every post is a measurement, not a promise:

- **Show the failure.** A wash is more interesting than a hold; it is the reason the band exists.
- **Name the band.** "Holds at 0.5-2.8, washes at 5.0" is the post.
- **No semantic claim.** Say "direction" and "strength", not "it understands fog".
- **No client renders.** The KCA production is not Helm and never appears.
- **One CTA.** The atlas or the repo, not both.

## Formats

### A — Axis card (default)

The atlas thumbnail is already a 4-up: three norm strengths plus the single push render available. The numbers use the top push dose. Add a 10-second vertical loop with a slow zoom on the same four frames, and a caption:

> `fog density` — same prompt, same seed, three norm strengths plus one push render. It already washes at the first dose: edge falls to 34% of the baseline at the top push dose. The useful band is below the wash, not at the top.

### B — Before/after carousel

Three cards: low, mid, wash. One line each. The last card says what broke. This is the highest-save format because it teaches a failure mode.

### C — Negative result

A chart from `charts/` plus three lines: what we hoped, what happened, what we shipped instead. Use `04_cos_vs_survival.png`, `06_effect_size.png` or `07_prompt_mix.png`. These are the posts that separate the project from a prompt pack.

### D — 60-second explainer

One axis, one scene, the slider in real time, the wash at the end, the band on screen. Reuse the loops and the existing H3 renders; no GPU window needed.

## First four weeks

| week | A | B |
|---|---|---|
| 1 | artificial light bursts (washes, onset 1.82) | backlight rim (washes, onset 0.85) |
| 2 | fluorescent vs tungsten (washes) | high key vs low key (washes, onset 0.32) |
| 3 | lens flare intensity (washes, onset 3.13) | light source position (washes, onset 3.38) |
| 4 | mixed color sources (washes, non-monotone) | night practicals (washes, onset 1.00) |

Start with the washes. The first month teaches the failure taxonomy; the holds land later, when the audience knows what a wash is.

## The five specials

1. **Three hypotheses we killed** — [`NEGATIVE-RESULTS.md`](NEGATIVE-RESULTS.md). The predictor, the probe estimator, norm/clamp.
2. **Helm vs prompt mixing** — `charts/07_prompt_mix.png`. A step is not a slider.
3. **Which models are measured?** — `charts/09_model_matrix.png`. Two measured, five not.
4. **The axis atlas** — `atlas/index.html`. 55 directions, one page.
5. **How this study is built** — [`DATASET.md`](DATASET.md). From 385 render jobs to a table.

## Cadence and channels

- **Two axis posts per week**, Tuesday and Friday. One special per month.
- **Instagram / TikTok:** format A, 9:16, 10-15 s, no voice, captions.
- **X:** format C or a still from A; the chart is the post.
- **LinkedIn:** format C with the method paragraph; the agency audience cares about the process.
- **YouTube Shorts:** format A or D.
- **Reddit (r/StableDiffusion, r/comfyui):** the negative results and the atlas, not the product. Read the self-promotion rules first.

## What to measure

Not likes. Track:

- saves and shares (the post taught something);
- atlas visits and the axes people search for;
- repo stars and issues that name a specific axis;
- replies that ask "does it work on X model?" — those are the next measurement.

## What not to do

- Do not post a hold without the wash next to it.
- Do not call Krea 2 or LTX measured.
- Do not use the client production as a demo.
- Do not make a "day 1 of 100" series; the atlas is finite and the honesty is the brand.
- Do not post the same four images without the caption's numbers.

## Repurpose map

| source | post |
|---|---|
| `atlas/thumbs/*.webp` | format A, 53 posts |
| `charts/04_cos_vs_survival.png` | the predictor thread |
| `charts/06_effect_size.png` | "a hold is not a big effect" |
| `charts/07_prompt_mix.png` | "a step is not a slider" |
| `charts/09_model_matrix.png` | "what is actually measured" |
| `data/axes.csv` | a table screenshot with one row highlighted |
| `NEGATIVE-RESULTS.md` | a three-post thread |
| `DATASET.md` | the method post |

The calendar is a starting order, not a contract. If an axis gets a strong reply, move it up; the data is already measured.
