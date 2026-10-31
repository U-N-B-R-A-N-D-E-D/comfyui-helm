from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    import os

    from aiohttp import web
    from server import PromptServer

    from . import nodes as _nodes

    @PromptServer.instance.routes.get("/helm/axes")
    async def _helm_axes(request):
        axes = []
        seen = set()
        for d in _nodes._READ_DIRS:
            try:
                entries = os.listdir(d)
            except OSError:
                continue
            for ext in (_nodes.AXIS_EXT, _nodes.LEGACY_EXT):
                for fn in entries:
                    if not fn.endswith(ext):
                        continue
                    name = fn[: -len(ext)]
                    if name in seen:
                        continue
                    seen.add(name)
                    meta = _nodes._read_meta(name) or {}
                    axes.append({
                        "name": name,
                        "path": os.path.join(d, fn),
                        "kind": meta.get("kind"),
                        "dim": meta.get("dim"),
                        "block": meta.get("block"),
                        "norm": meta.get("norm"),
                        "safe_max": meta.get("safe_max"),
                        "created": meta.get("created"),
                        "tested": meta.get("tested"),
                    })
        # `dial` is the menu the HelmDial node offers: Essentials (benchmarked) then Custom.
        pkg = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        dial = _nodes._dial_entries()
        for e in dial:  # a real before/after for the axes that have one, served from this pack's web folder
            rel = (_nodes._catalog().get(e["file"]) or {}).get("preview")
            if rel and e["section"] == _nodes.ESSENTIALS:
                e["preview"] = f"/extensions/{pkg}/{rel}"
        return web.json_response({"axes": axes, "dial": dial,
                                  "dirs": list(_nodes._READ_DIRS)})
except Exception:
    pass
