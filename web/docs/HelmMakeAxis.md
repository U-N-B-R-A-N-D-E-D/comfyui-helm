# Helm · Make Axis

Make your own idea for the Dial from two prompts.

## Use it

1. Write **with_idea**: a prompt that has the idea (for example *thick fog*).
2. Write **without_idea**: the same prompt without it (*clear air*). Change only the idea.
3. Give it a **name** and run.

The idea is saved in your ComfyUI user folder and shows up in the Dial menu under **Custom**. Connect the `axis` output to the Dial's `axis_from` input to use it right away.

## Good to know

- Use a small latent (512) for the test renders. They are thrown away.
- The safe range of a new axis is an estimate, not a measurement. Compare Before | After and lower Amount if the picture falls apart.
- If nothing changes, make the two prompts differ more.
