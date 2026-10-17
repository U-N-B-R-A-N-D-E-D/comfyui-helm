"""Headless test harness for the Helm node pack.

Stubs `folder_paths` and `comfy.memory_management` before importing `nodes.py`, then exposes a
`helm` fixture pointing the axis read/write dirs at `tmp_path`. No running ComfyUI is required; the
tests use real torch (CPU).
"""
import importlib.util
import os
import sys
import tempfile
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_USER_DIR = tempfile.mkdtemp(prefix="helm-user-")
_MODELS_DIR = tempfile.mkdtemp(prefix="helm-models-")


def _ensure_stubs():
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_user_directory = lambda: _USER_DIR
        fp.models_dir = _MODELS_DIR
        fp.get_folder_paths = lambda *a, **k: []
        sys.modules["folder_paths"] = fp

    if "comfy.memory_management" not in sys.modules:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        mm = types.ModuleType("comfy.memory_management")
        mm.aimdo_enabled = True
        comfy.memory_management = mm
        sys.modules["comfy"] = comfy
        sys.modules["comfy.memory_management"] = mm

    if "comfy_sdk" not in sys.modules:
        sdk = types.ModuleType("comfy_sdk")

        class _NoNetworkComfy:
            def __init__(self, *a, **k):
                raise AssertionError("Comfy() network client constructed during a headless test")

        sdk.Comfy = _NoNetworkComfy
        sys.modules["comfy_sdk"] = sdk


_ensure_stubs()

_NODES = None


def import_nodes():
    global _NODES
    if _NODES is None:
        _ensure_stubs()
        path = os.path.join(ROOT, "nodes.py")
        spec = importlib.util.spec_from_file_location("helm_nodes", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["helm_nodes"] = mod
        spec.loader.exec_module(mod)
        _NODES = mod
    return _NODES


# --- shared fakes / helpers (importable from tests) ---------------------------------------

def deepcopy_list_dict(obj, memo=None):
    """Mirror comfy.utils.deepcopy_list_dict: recurse dicts/lists, everything else by reference."""
    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    if isinstance(obj, dict):
        res = {deepcopy_list_dict(k, memo): deepcopy_list_dict(v, memo) for k, v in obj.items()}
    elif isinstance(obj, list):
        res = [deepcopy_list_dict(i, memo) for i in obj]
    else:
        res = obj
    memo[oid] = res
    return res


class FakePatcher:
    """Minimal stand-in for ComfyUI's ModelPatcher (clone copies model_options like the real one)."""

    def __init__(self, model, model_options=None):
        self.model = model
        self.model_options = {} if model_options is None else model_options

    def clone(self):
        return FakePatcher(self.model, deepcopy_list_dict(self.model_options))

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


class FakeDiT(torch.nn.Module):
    def __init__(self, dim=8, n=6):
        super().__init__()
        self.dim = dim
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def make_dit(dim=8, n=6):
    return FakeDiT(dim=dim, n=n)


def fake_model(dit):
    """A patcher-like object: `_find_blocks` reads `.model.diffusion_model`."""
    return FakePatcher(types.SimpleNamespace(diffusion_model=dit))


def make_apply_model(dit):
    def apply_model(x, t, **kwargs):
        return dit(x)

    return apply_model


def make_c_in(x, sig=None, flags=None):
    if flags is None:
        flags = [0] * x.shape[0]
    to = {}
    if sig is not None:
        to["sample_sigmas"] = torch.tensor(sig, dtype=torch.float32)
    return {
        "input": x,
        "timestep": torch.zeros(1),
        "c": {"transformer_options": to},
        "cond_or_uncond": flags,
    }


@pytest.fixture
def helm(tmp_path, monkeypatch):
    nodes = import_nodes()
    user = tmp_path / "user"
    models = tmp_path / "models"
    steer = user / "steer"
    models_steer = models / "steer"
    for d in (steer, models_steer):
        d.mkdir(parents=True, exist_ok=True)

    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(user))
    monkeypatch.setattr(fp, "models_dir", str(models))
    if hasattr(nodes, "AXIS_DIR"):
        monkeypatch.setattr(nodes, "AXIS_DIR", str(steer))
    if hasattr(nodes, "_READ_DIRS"):
        pack_examples = os.path.join(os.path.dirname(os.path.realpath(nodes.__file__)), "example_axes")
        monkeypatch.setattr(nodes, "_READ_DIRS", [str(steer), str(models_steer), pack_examples])

    return types.SimpleNamespace(
        nodes=nodes,
        user=str(user),
        steer=str(steer),
        models=str(models),
        models_steer=str(models_steer),
        pack=os.path.dirname(os.path.realpath(nodes.__file__)),
        root=ROOT,
    )
