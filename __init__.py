from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

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
            for fn in entries:
                if not fn.endswith(".pt"):
                    continue
                name = fn[:-3]
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
        return web.json_response({"axes": axes, "dirs": list(_nodes._READ_DIRS)})
except Exception:
    pass
