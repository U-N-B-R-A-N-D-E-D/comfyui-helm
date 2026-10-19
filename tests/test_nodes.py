import torch
import pytest


def test_helm_vector_refuses_mismatched_blocks(helm):
    helm.nodes._save_axis("p", torch.ones(6), {"kind": "axis", "dim": 6, "block": 20, "model": "ZImage"})
    helm.nodes._save_axis("q", torch.ones(6), {"kind": "axis", "dim": 6, "block": 12, "model": "ZImage"})
    with pytest.raises(ValueError):
        helm.nodes.HelmVector().run("p", "q", "axis")
