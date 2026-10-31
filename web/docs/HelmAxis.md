# Helm · Axis

**What** idea to move. Pick one from the library, or build a new one from your own prompts.

## Use it

1. Pick an **idea**, for example *Light angle (raking)*.
2. Connect the **helm** output to **Helm · Apply**.
3. Run. That is the whole minimum: Axis, then Apply.

## Sources

- **library**: a ready-made idea. This is all you see in this mode.
- **prompts**: build a new idea from pairs. More settings appear.

## Build from prompts

Write one pair per line: `with the idea || without the idea`.

```
a quiet street at dawn, thick fog || a quiet street at dawn, clear air
a mountain lake, thick fog || a mountain lake, clear air
```

- Change **only the idea** inside a pair, and use **different subjects** in different pairs. What the pairs share is the idea.
- Connect **model**, **clip** and **latent** (a small empty latent is enough).
- **debias** (experimental) removes the colour tint that any prompt change tends to bring.
- Your idea is saved under **Custom** in the idea menu.

## Chain

Connect another Helm node to **helm** to add this idea to the ones before it.

The name in brackets in the menu is the model the idea was made for. Pick one that matches your model.
