"""Guard the Registry package against dev-only files and dynamic-import tripwires."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_comfyignore_excludes_dev_paths():
    ignore = (ROOT / ".comfyignore").read_text()
    for path in ("tests/", "docs/", "study/", ".github/", "examples/"):
        assert path in ignore, f"{path} must stay out of the Registry package"


def test_runtime_has_no_dynamic_import():
    for name in ("__init__.py", "nodes.py", "app/run_helm.py"):
        text = (ROOT / name).read_text()
        assert "__import__(" not in text, f"dynamic import in shipped file: {name}"
