#!/usr/bin/env python3
"""Run a Helm workflow through the Comfy SDK (Comfy Developer Platform).

The same code runs against a self-hosted ComfyUI (behind comfy-api-proxy), Comfy Cloud, or a
serverless deployment -- only COMFY_BASE_URL (and an API key, for the hosted surfaces) change.

Self-hosted ComfyUI with this node pack installed:
    pip install comfy-sdk
    pip install comfy-api-proxy
    comfy-api-proxy                                  # serves Comfy API v2 on 127.0.0.1:8189 in front of ComfyUI :8188
    export COMFY_BASE_URL=http://127.0.0.1:8189
    python app/run_helm.py --strength 0.2 --out out

Comfy Cloud / serverless:
    export COMFY_BASE_URL=https://<deployment>.run.comfy.app
    export COMFY_API_KEY=comfyui-...
    python app/run_helm.py --strength 0.2 --out out

Per-node control:
    --strength is a global default applied to every HelmApply node; --axis globally overrides the
    axis name. Use a repeatable --set <axis>=<strength> to target individual nodes (matched by their
    current axis name, or by graph position as a fallback). --list prints every HelmApply node
    (index, axis, strength, block) in graph order and exits:
    python app/run_helm.py --set mood=0.15 --set density=0.25 --out out
    python app/run_helm.py --list
"""
import argparse
import json
import os


def parse_sets(sets):
    parsed = []
    for item in sets or []:
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise SystemExit(f"--set expects <axis>=<strength>, got '{item}'")
        try:
            strength = float(value)
        except ValueError:
            raise SystemExit(f"--set strength must be a number, got '{item}'")
        parsed.append((name, strength))
    return parsed


def _match_node(name, nodes):
    for i, (_, node) in enumerate(nodes):
        if node.get("inputs", {}).get("axis") == name:
            return i
    try:
        pos = int(name)
    except ValueError:
        return None
    if 0 <= pos < len(nodes):
        return pos
    return None


def build_plan(wf_json, strength, axis, sets):
    nodes = [
        (node_id, node)
        for node_id, node in wf_json.items()
        if isinstance(node, dict) and node.get("class_type") == "HelmApply"
    ]
    if not nodes:
        raise SystemExit("no HelmApply node in the workflow")

    overrides = {}
    matched = set()
    for name, value in parse_sets(sets):
        index = _match_node(name, nodes)
        if index is None:
            found = ", ".join(
                f"{i}:{n.get('inputs', {}).get('axis')}" for i, (_, n) in enumerate(nodes)
            )
            raise SystemExit(f"--set '{name}' matched no HelmApply node; found {found}")
        overrides[index] = value
        matched.add(index)

    plan = []
    for i, (node_id, node) in enumerate(nodes):
        inputs = node.get("inputs", {})
        current_axis = inputs.get("axis")
        plan.append(
            {
                "index": i,
                "axis": current_axis if i in matched else (axis or current_axis),
                "strength": overrides.get(i, strength),
                "block": inputs.get("block"),
                "node_id": node_id,
            }
        )
    return plan


def format_line(entry):
    return f"{entry['index']}  {entry['axis']}  {entry['strength']}  [{entry['block']}]"


def main():
    ap = argparse.ArgumentParser(description="Run a Helm workflow via the Comfy SDK.")
    ap.add_argument("--workflow", default="api/h3_t2v_helm_api.json")
    ap.add_argument("--strength", type=float, default=0.2,
                    help="global default Helm slider value for every HelmApply node (see calibration.json)")
    ap.add_argument("--axis", default=None,
                    help="global override of the axis name on HelmApply nodes not matched by --set")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="AXIS=STRENGTH",
                    help="set one HelmApply node's strength by axis name (or graph position); repeatable")
    ap.add_argument("--list", dest="list_only", action="store_true",
                    help="print each HelmApply node (index, axis, strength, block) in graph order and exit")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    with open(args.workflow) as f:
        wf_json = json.load(f)

    plan = build_plan(wf_json, args.strength, args.axis, args.sets)

    if args.list_only:
        for entry in plan:
            print(format_line(entry))
        return

    os.makedirs(args.out, exist_ok=True)
    from comfy_sdk import Comfy

    client = Comfy()  # base URL from COMFY_BASE_URL; no key needed for self-hosted
    wf = client.workflows.from_file(args.workflow)

    for entry in plan:
        wf.set_input(entry["node_id"], "strength", entry["strength"])
        if entry["axis"]:
            wf.set_input(entry["node_id"], "axis", entry["axis"])

    sliders = ", ".join(f"{e['axis']}={e['strength']}" for e in plan)
    print(f"submitting {args.workflow} ({len(plan)} Helm slider(s), sliders={sliders})")
    job = client.run(wf)
    for out in job.outputs:
        if out.type in ("video", "image", "audio"):
            path = os.path.join(args.out, out.name)
            out.to_file(path)
            print("wrote", path, out.type)


if __name__ == "__main__":
    main()
