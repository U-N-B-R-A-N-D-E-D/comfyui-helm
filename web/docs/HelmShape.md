# Helm · Shape (experimental)

**How much** and **when**. Optional: Axis can go straight to Apply.

## Use it

1. Put Shape after Axis (or after Mix).
2. Move **amount**. The line under the slider says it in words, for example *−40% (less), late steps*.
3. Connect the result to **Helm · Apply**.

| Amount | What you get |
|:--|:--|
| 0 | Off. |
| +1 | The strongest setting that still looks right for this idea. |
| −1 | The opposite idea, at the same strength. |

Shape sets the amount for every idea already in the chain.

## More settings (they appear when you need them)

- **stage**: *early steps* shape the composition, *late steps* the finish. **split** says where early ends.
- **curve** (video, MiniMax H3): *ramp* goes from **amount** on the first frame to **end_amount** on the last. *keyframes* uses your own list, `position:amount`, for example `0:0.2, 0.5:0.8, 1:0.4`.

Stage and curve are experimental and not yet measured on a GPU.
