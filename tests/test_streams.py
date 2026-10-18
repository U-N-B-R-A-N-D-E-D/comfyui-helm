"""A block that returns a tuple must declare which element is the visual stream.

The bug this pins: `out[0]` is the image only when the tuple happens to be (img, txt). Flux is
(layers.py:267); SD3 and Qwen-Image are (txt, img), so assuming out[0] would steer the TEXT
conditioning. Helm now refuses an undeclared tuple block instead of guessing.
"""
import torch
import pytest

from conftest import FakePatcher


class DoubleStreamBlock(torch.nn.Module):
    """Named like the real Flux block, whose forward returns (img, txt)."""

    def forward(self, x):
        return x, torch.zeros_like(x)


class JointBlock(torch.nn.Module):
    """SD3/Qwen shape: (txt, img). Unknown to the table on purpose."""

    def forward(self, x):
        return torch.zeros_like(x), x


def test_flux_style_tuple_steers_the_image_not_the_text(helm):
    fn = helm.nodes._make_apply_hook(torch.ones(8), 0.5, "ax", False, "meta")
    img, txt = torch.zeros(1, 8), torch.full((1, 8), 7.0)
    out = fn(DoubleStreamBlock(), (), (img, txt))
    assert torch.allclose(out[0], torch.full((1, 8), 0.5)), "the axis did not reach the image stream"
    assert torch.allclose(out[1], txt), "the text stream was modified"


def test_an_undeclared_tuple_block_is_refused_not_guessed(helm):
    fn = helm.nodes._make_apply_hook(torch.ones(8), 0.5, "ax", False, "meta")
    with pytest.raises(RuntimeError) as ei:
        fn(JointBlock(), (), (torch.zeros(1, 8), torch.zeros(1, 8)))
    assert "visual stream" in str(ei.value)


def test_a_plain_tensor_block_is_untouched(helm):
    fn = helm.nodes._make_apply_hook(torch.ones(8), 0.5, "ax", False, "meta")
    out = fn(torch.nn.Identity(), (), torch.zeros(1, 8))
    assert not isinstance(out, tuple)
    assert torch.allclose(out, torch.full((1, 8), 0.5))
