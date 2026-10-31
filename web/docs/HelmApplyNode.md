# Helm · Apply

**Where** the ideas go. This is the only 0.5 node that changes the model.

## Use it

1. Put Apply between your model loader and the sampler.
2. Connect **helm** from Axis, Shape or Mix.
3. Connect the **model** output to your sampler.

Minimum setup: **Helm · Axis → Helm · Apply**. Without a Shape the default is amount 0.5 on every step.

## Settings

- **cond_only**: off by default. Add the idea only to the prompt half of each guided step.
- **mode**: *push* adds the idea (default). *norm* rescales to each token's size. *clamp* never goes past what the model already makes.

If every amount or weight is 0, Apply returns the model untouched.

An idea made for another model is refused with a plain message.
