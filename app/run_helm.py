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
"""
import argparse
import os

from comfy_sdk import Comfy


def main():
    ap = argparse.ArgumentParser(description="Run a Helm workflow via the Comfy SDK.")
    ap.add_argument("--workflow", default="api/h3_t2v_helm_api.json")
    ap.add_argument("--strength", type=float, default=0.2, help="Helm slider value (see calibration.json)")
    ap.add_argument("--axis", default=None, help="override the axis name on every HelmApply node")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = Comfy()  # base URL from COMFY_BASE_URL; no key needed for self-hosted
    wf = client.workflows.from_file(args.workflow)

    sliders = 0
    for node_id, node in wf.json.items():
        if node.get("class_type") == "HelmApply":
            wf.set_input(node_id, "strength", args.strength)
            if args.axis:
                wf.set_input(node_id, "axis", args.axis)
            sliders += 1
    if not sliders:
        raise SystemExit("no HelmApply node in the workflow")

    print(f"submitting {args.workflow} ({sliders} Helm slider(s), strength={args.strength})")
    job = client.run(wf)
    for out in job.outputs:
        if out.type in ("video", "image", "audio"):
            path = os.path.join(args.out, out.name)
            out.to_file(path)
            print("wrote", path, out.type)


if __name__ == "__main__":
    main()
