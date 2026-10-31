"""web/helm_flow.js: the "what comes next" suggestions match the chain and name only nodes that exist."""
import os
import re

PACK = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _js():
    return open(os.path.join(PACK, "web", "helm_flow.js"), encoding="utf-8").read()


def _names(row):
    return re.findall(r'"(\w+)"', row)


def test_suggestions_after_each_helm_output():
    js = _js()
    expected = {
        "HelmAxis": ["HelmShape", "HelmMix", "HelmApplyNode", "HelmAxis"],
        "HelmShape": ["HelmApplyNode", "HelmMix"],
        "HelmMix": ["HelmApplyNode"],
    }
    for node, names in expected.items():
        m = re.search(node + r": \{ HELM: \[([^\]]*)\] \}", js)
        assert m and _names(m.group(1)) == names, node
    assert re.search(r"HelmApplyNode: \{ MODEL: \[\"KSampler\", \"BasicGuider\"", js)
    assert re.search(r"const BEFORE = \{ HELM: \[\"HelmAxis\"\] \};", js), "a HELM input suggests Axis"


def test_suggested_helm_nodes_exist(helm):
    for cls in set(re.findall(r'"(Helm\w+)"', _js())):
        assert cls in helm.nodes.NODE_CLASS_MAPPINGS, cls


def test_override_is_scoped_to_helm_nodes_and_restored():
    js = _js()
    assert "restoreTables" in js and "MutationObserver" in js
    assert 'startsWith("Helm")' in js, "inputs are only touched on Helm nodes"
