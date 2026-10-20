"""The three steering modes and the mask layout helper.

These are pure-function tests: no model, no ComfyUI. They pin the math that the evolution
window measures, so a regression cannot silently change what "norm" or "clamp" means.
"""
import torch

import nodes


def test_push_is_exactly_the_published_behavior():
    x = torch.randn(2, 5, 8)
    vec = torch.randn(8)
    d = vec * 0.3
    y = nodes._steer(x, d, None, mode="push")
    assert torch.equal(y, x + d)


def test_norm_preserves_each_token_norm():
    x = torch.randn(2, 5, 8)
    vec = torch.randn(8)
    y = nodes._steer(x, vec * 2.0, None, mode="norm")
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    assert torch.allclose(nx, ny, atol=1e-5)
    assert not torch.allclose(x, y)  # it still moved, just without the energy


def test_clamp_never_leaves_the_norm_budget():
    x = torch.randn(1, 4, 8)
    u = torch.randn(8)
    u = u / u.norm()
    huge = u * 1e6
    y = nodes._steer(x, huge, u, mode="clamp", clamp_k=1.0)
    p_new = (y * u).sum(dim=-1)
    lim = x.norm(dim=-1)
    assert torch.all(p_new.abs() <= lim + 1e-4)


def test_clamp_with_small_delta_matches_push_along_the_axis():
    x = torch.randn(1, 3, 8)
    u = torch.randn(8)
    u = u / u.norm()
    d = u * 0.01
    y_clamp = nodes._steer(x, d, u, mode="clamp", clamp_k=10.0)
    y_push = nodes._steer(x, d, None, mode="push")
    assert torch.allclose(y_clamp, y_push, atol=1e-5)


def test_unknown_mode_fails_loudly():
    x = torch.zeros(1, 2, 4)
    try:
        nodes._steer(x, x, None, mode="warp")
    except ValueError as e:
        assert "warp" in str(e)
    else:
        raise AssertionError("unknown mode must raise")


def test_parse_grid():
    assert nodes._parse_grid("64,64") == (64, 64, 1)
    assert nodes._parse_grid("8,12,5") == (8, 12, 5)
    for bad in ("64", "1,2,3,4", ""):
        try:
            nodes._parse_grid(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise")


def test_token_mask_layouts():
    m = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    flat = nodes._token_mask(m, "h_w", 1).reshape(-1).tolist()
    assert flat == [1, 0, 0, 1]
    # t_h_w: each frame carries the same 2x2 mask
    flat = nodes._token_mask(m, "t_h_w", 2).reshape(-1).tolist()
    assert flat == [1, 0, 0, 1, 1, 0, 0, 1]
    # h_w_t: t is the inner axis, so each (h,w) position repeats per frame
    flat = nodes._token_mask(m, "h_w_t", 2).reshape(-1).tolist()
    assert flat == [1, 1, 0, 0, 0, 0, 1, 1]


def test_hook_applies_the_mask_to_the_delta_only():
    x = torch.zeros(1, 4, 8)
    vec = torch.ones(8)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    fn = nodes._make_apply_hook(vec, 1.0, "t", False, "meta", mode="push",
                                token_mask=(mask, "h_w", 1))
    y = fn(object(), (), x)
    assert y[0, 0].tolist() == [1.0] * 8
    assert y[0, 1].tolist() == [0.0] * 8
    assert y[0, 2].tolist() == [1.0] * 8
    assert y[0, 3].tolist() == [0.0] * 8


def test_mask_offset_skips_the_leading_packed_tokens():
    x = torch.zeros(1, 6, 4)
    vec = torch.ones(4)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    fn = nodes._make_apply_hook(vec, 1.0, "t", False, "meta", mode="push",
                                token_mask=(mask, "h_w", 1), mask_offset=2)
    y = fn(object(), (), x)
    assert y[0, 0].tolist() == [0.0] * 4
    assert y[0, 1].tolist() == [0.0] * 4
    assert y[0, 2].tolist() == [1.0] * 4
    assert y[0, 3].tolist() == [0.0] * 4
    assert y[0, 4].tolist() == [1.0] * 4
    assert y[0, 5].tolist() == [0.0] * 4


def test_mask_offset_on_a_packed_2d_block():
    x = torch.zeros(5, 3)
    vec = torch.ones(3)
    mask = torch.tensor([[1.0, 0.0]])
    fn = nodes._make_apply_hook(vec, 1.0, "t", False, "meta", mode="push",
                                token_mask=(mask, "h_w", 1), mask_offset=3)
    y = fn(object(), (), x)
    assert y.shape == x.shape
    assert y[3].tolist() == [1.0, 1.0, 1.0]
    assert y[4].tolist() == [0.0, 0.0, 0.0]


def test_mask_offset_mismatch_is_refused():
    x = torch.zeros(1, 6, 4)
    vec = torch.ones(4)
    mask = torch.tensor([[1.0, 0.0]])
    fn = nodes._make_apply_hook(vec, 1.0, "t", False, "meta", mode="push",
                                token_mask=(mask, "h_w", 1), mask_offset=2)
    try:
        fn(object(), (), x)
    except RuntimeError as e:
        assert "mask covers 2 tokens + offset 2 = 4" in str(e)
    else:
        raise AssertionError("a mismatched mask_offset must be refused")
