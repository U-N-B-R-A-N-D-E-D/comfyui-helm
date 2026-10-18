"""cond_only masking for both the collect mean and the apply add (spec 1.5 / 1.6)."""
import torch

import conftest


def test_mask_by_flags_zeroes_uncond_rows():
    nodes = conftest.import_nodes()
    x = torch.zeros(2, 3)
    d = torch.ones(3)
    masked = nodes._mask_by_flags(d, x, [0, 1])
    assert torch.allclose(masked[0], torch.ones(3))
    assert torch.allclose(masked[1], torch.zeros(3))


def test_mask_by_flags_length_mismatch_is_permissive():
    nodes = conftest.import_nodes()
    x = torch.zeros(3, 3)
    d = torch.ones(3)
    masked = nodes._mask_by_flags(d, x, [0, 1])
    assert masked.shape[-1] == 3
    assert torch.isfinite(masked).all()


def test_stack_mean_cond_only_selects_flag_zero():
    nodes = conftest.import_nodes()
    out = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[9.0, 9.0], [9.0, 9.0]]])
    m = nodes._stack_mean(torch.nn.Identity(), out, flags=[0, 1], cond_only=True)
    assert torch.allclose(m, torch.ones(2))


def test_stack_mean_all_items_when_not_cond_only():
    nodes = conftest.import_nodes()
    out = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[9.0, 9.0], [9.0, 9.0]]])
    m = nodes._stack_mean(torch.nn.Identity(), out, flags=[0, 1], cond_only=False)
    assert torch.allclose(m, torch.full((2,), 5.0))


def test_apply_hook_cond_only_masks_uncond():
    nodes = conftest.import_nodes()
    fn = nodes._make_apply_hook(torch.ones(3), 1.0, "ax", True, "meta")
    x = torch.zeros(2, 3)
    y = fn(None, (), x, flags=[0, 1])
    y = y[0] if isinstance(y, tuple) else y
    assert torch.allclose(y[0], torch.ones(3))
    assert torch.allclose(y[1], torch.zeros(3))


def test_apply_hook_cond_only_false_adds_to_all():
    nodes = conftest.import_nodes()
    fn = nodes._make_apply_hook(torch.ones(3), 1.0, "ax", False, "meta")
    x = torch.zeros(2, 3)
    y = fn(None, (), x, flags=[0, 1])
    y = y[0] if isinstance(y, tuple) else y
    assert torch.allclose(y, torch.ones(2, 3))
