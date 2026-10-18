

def test_helm_vector_refuses_mismatched_blocks(helm):
    helm.nodes._save_axis("p", torch.ones(6), {"kind": "axis", "dim": 6, "block": 20, "model": "ZImage"})
    helm.nodes._save_axis("q", torch.ones(6), {"kind": "axis", "dim": 6, "block": 12, "model": "ZImage"})
    with pytest.raises(ValueError):
        helm.nodes.HelmVector().run("p", "q", "axis")


def test_ambiguous_block_lists_are_named_not_guessed(helm):
    class DM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)])
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.diffusion_model = DM()
    with pytest.raises(RuntimeError) as ei:
        helm.nodes._find_blocks(M())
    assert "ambiguous" in str(ei.value) and "layers[6]" in str(ei.value)


def test_cond_only_without_flags_refuses(helm):
    with pytest.raises(RuntimeError):
        helm.nodes._stack_mean(torch.nn.Identity(), torch.ones(2, 6), flags=None, cond_only=True)
