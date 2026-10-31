# Helm · Dial

Move one idea with one slider. Your prompt is not rewritten.

## Use it

1. Put the Dial between your model loader and the sampler.
2. Pick an **axis** (the idea), for example *Light angle (raking)*.
3. Set **Amount** and run.

| Amount | What you get |
|:--|:--|
| 0 | Helm is off. The picture is exactly what the model would draw without it. |
| 0.5 | Half of the safe range. A good place to start. |
| 1 | The strongest setting that still looks right for this idea. |

The line under the slider says what the Dial is doing, for example *Light angle (raking) at 60%*.

## The menu

- **Essentials**: ideas Helm's own benchmark shows still look right when moved, listed by how visible the change is.
- **Subtle**: ideas that hold up too, but change the picture only a little.
- **Custom**: axes you made with *Helm · Make Axis* or dropped into your steer folder. Their safe range is an estimate.

The name in brackets is the model the idea was made for. Pick one that matches the model you loaded.

## Compare

Render twice with the same seed, once with Amount 0 and once with your Amount, and compare. The starter templates do this for you.
