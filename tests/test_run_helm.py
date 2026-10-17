"""`app/run_helm.py` plan/CLI surface: build_plan exposure and `--list` (spec 2).

The app agent owns run_helm.py and is refactoring it in parallel; tests that need a not-yet-exposed
`build_plan` skip gracefully so the rest of the suite still reports.
"""
import importlib.util
import os
import sys

import pytest

import conftest

ROOT = conftest.ROOT


def _load_run_helm():
    path = os.path.join(ROOT, "app", "run_helm.py")
    spec = importlib.util.spec_from_file_location("helm_run", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["helm_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_main(mod, argv):
    old = sys.argv
    sys.argv = ["run_helm.py"] + argv
    code = 0
    try:
        result = mod.main()
        if isinstance(result, int):
            code = result
    except SystemExit as e:
        code = 0 if e.code is None else e.code
    finally:
        sys.argv = old
    return code


def test_build_plan_exposed():
    mod = _load_run_helm()
    if not hasattr(mod, "build_plan"):
        pytest.skip("build_plan not yet exposed by agent-app (in flight)")
    assert callable(mod.build_plan)


def test_cli_list_lists_helmapply_in_graph_order(helm, tmp_path, capsys):
    mod = _load_run_helm()
    api = os.path.join(ROOT, "api", "h3_t2v_helm_api.json")
    code = _run_main(mod, ["--workflow", api, "--out", str(tmp_path / "out"), "--list"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code in (0, None), f"--list exited {code!r}: {out}"
    assert "mood" in out and "density" in out, f"--list did not list both HelmApply axes: {out!r}"
    assert out.index("mood") < out.index("density"), "HelmApply nodes are not listed in graph order"


def test_cli_unknown_set_errors_and_names_found(helm, tmp_path, capsys):
    mod = _load_run_helm()
    api = os.path.join(ROOT, "api", "h3_t2v_helm_api.json")
    with pytest.raises(SystemExit) as ei:
        _run_main_raise(mod, ["--workflow", api, "--out", str(tmp_path / "out"), "--set", "nope=0.5"])
    assert ei.value.code not in (0, None)
    message = str(ei.value) + capsys.readouterr().out
    assert "mood" in message, f"error did not list the found HelmApply nodes: {message!r}"


def _run_main_raise(mod, argv):
    old = sys.argv
    sys.argv = ["run_helm.py"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
