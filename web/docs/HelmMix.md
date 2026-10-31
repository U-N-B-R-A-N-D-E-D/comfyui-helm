# Helm · Mix (experimental)

**Combine** ideas, so you move several at once.

## Use it

1. Connect up to four Axis nodes (or chains) to **helm_1** to **helm_4**.
2. Set a **weight** from 0 to 1 for each one.
3. Connect **helm** to Shape or Apply.

The line under the title tells you what Mix did. After a run it shows the report, including warnings.

## Orthogonalize

On by default. Mix makes the active ideas independent of each other, in order, so moving one does not drag another with it. The first idea stays as it is.

## Good to know

- Ideas should be for the same model and block. Mix warns when they are not.
- **axis_weights** is an optional list, one weight per idea in order, for example `1, 0.5, 0.8`.
- A weight never goes above 1, so no idea can leave its safe range.
- The **report** output has the same text as the line under the title.
