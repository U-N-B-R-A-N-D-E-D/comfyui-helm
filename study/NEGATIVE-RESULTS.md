# Three hypotheses we killed

A study is only as good as the things it refuses to ship. These are the three we built, measured and removed. The numbers are in [`data/axes.csv`](data/axes.csv), [`data/labels_long.csv`](data/labels_long.csv) and [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## 1. "Alignment with the mean activation screens the axes"

**The lead.** At n=23, the ones that survived sat close to the mean activation direction; the ones that collapsed sat far from it. It looked like a free screen: no renders, just a cosine. We wrote the criterion down before the next run: survivors above ~0.9, washes below ~0.7, and a single counter-example kills it.

**What happened at n=78.** The ranges overlapped. The lowest survivor is **0.310** (`saturated_vs_muted_stock`); the highest wash is **0.982** (`symmetry`). Every hold sits inside the wash range. A threshold that keeps the survivors also keeps most of the washes.

**Why it matters.** The first version of this note said "the screen holds (n=11)". It did hold at n=11. It did not hold at n=78. The lesson is not "cosine is useless"; it is that a separation measured on 11-23 points is a hypothesis, not a screen. We kept the numbers and deleted the claim.

**What would change our mind.** A threshold with zero false positives and zero false negatives on a held-out set of at least 20 axes, chosen before looking at the labels. Until then, the cosine is a description, not a filter.

## 2. "The probe can suggest a strength"

**The idea.** From the H3 numbers, a perturbation was roughly 30% of the block's activation norm across the verified band. So `strength ≈ 0.3 × |mu| / |dmu|`. Run `HelmProbe`, get a suggested strength, skip the manual sweep.

**What happened.** On Z-Image the formula returns **8.8** at the probe's peak. The band verified by rendering is **0.5-3.0**. An order of magnitude out. A number that wrong, under the word "suggested", is worse than no number.

**What we shipped instead.** `HelmProbeReport` returns the per-block separation profile and the middle-band peak **as a measurement**. It does not pick the block: the probe's highest-separation block inside the middle 50% is **7**, two blocks below the rendered band (9-21), and the global peak is block **0**, which destroys the frame.

**What would change our mind.** A calibration that predicts the wash onset within ±20% on a held-out model, with the residual reported. The probe is still useful for "where does the contrast live"; it is not a calibrator.

## 3. "Norm-preserving or projection-clamping fixes the wash"

**The idea.** The wash looks like a magnitude problem. If we rescale each token to its original norm (`norm`) or clamp the projection inside a `k·||x||` budget (`clamp`), the direction should survive more often than the raw push.

**What happened.** Under the same pre-registered criterion, on the same 55 dosed axes and the same three strengths:

| mode | holds | washes | unmeasured |
|---|---:|---:|---:|
| push | **10** | 43 | 2 |
| norm | **1** | 52 | 2 |
| clamp | **2** | 51 | 2 |

Norm and clamp are **worse**, not better. The mechanism is visible in the per-axis data: they collapse luminance (e.g. `aisle_width` from 0.453/0.248 to 0.072 at the top strength; `shadow_hardness` to 0.003) while `push` keeps the frame lit. Preserving the norm after adding a large vector does not remove the wash; it keeps the frame in the dark.

**What survives.** The step window is a partial mitigation, not a fix: applying `push` only in the last 40% moves the frame less and preserves structure better, but it does not rescue a bad direction. The usable lever is the strength band and the axis choice.

## What we did not ship

- The probe as a block picker (wrong block).
- The probe as a strength estimator (10× out).
- The cosine as a screen (overlap at 78).
- `norm` and `clamp` as defaults (1/53 and 2/53).
- The first prompt-mix test (it compared opposite directions; the fair re-test is in `EXPERIMENTS.md` §E).

## The method

1. Write the criterion before the run.
2. Measure the negative result with the same care as the positive one.
3. Publish the number that kills the hypothesis.
4. Keep the artefact if it is still useful for something narrower.

The study layer exists so those numbers are a table, not a paragraph. If you want to challenge one, the CSVs are the place to start.
