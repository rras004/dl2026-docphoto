"""Checks for the lab-4 tasks. Each fails with NotImplementedError until its task is written.

    docker compose run --rm dev pytest tests/test_lab4.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from src import optim, sanity
from src.data import augment
from src.losses import MSE
from src.models import cnn


def small_net(**kw):
    torch.manual_seed(0)
    return cnn.SmallCNN(in_ch=1, widths=[8, 16, 32], prior=0.8, **kw)


def test_smallcnn_returns_one_logit_per_image() -> None:
    net = small_net().eval()
    with torch.no_grad():
        out = net(torch.randn(4, 1, 64, 64))
    assert out.shape == (4,), "forward must return shape (batch,): squeeze(1) after the head"
    assert hasattr(net, "features") and hasattr(net, "head"), "name the parts self.features and self.head"
    assert torch.sigmoid(out).sub(0.8).abs().max() < 0.05, "the untrained network should predict the prior"


def test_smallcnn_accepts_other_sizes_and_has_no_sigmoid() -> None:
    net = small_net().eval()
    with torch.no_grad():
        a = net(torch.randn(2, 1, 96, 64))
    assert a.shape == (2,)
    assert float(a.max()) > 1.0, "the output is a logit (about logit(0.8) = 1.39), not a probability"


def test_overfit_one_batch_reaches_nearly_zero() -> None:
    torch.manual_seed(0)
    xb, yb = torch.randn(8, 1, 32, 32), torch.rand(8)
    hist = sanity.overfit_one_batch(small_net(), xb, yb, MSE(prior=0.5), steps=150, lr=3e-3)
    assert len(hist) == 150
    assert hist[-1] < 1e-3 < hist[0], f"loss went from {hist[0]:.4f} to {hist[-1]:.4f}"


def test_same_seed_twice_is_identical() -> None:
    xb, yb = torch.randn(4, 1, 32, 32), torch.rand(4)
    a, b = sanity.same_seed_twice(small_net, xb, yb, MSE(prior=0.5), seed=3, steps=3)
    assert len(a) == len(b) == 3
    assert np.allclose(a, b, rtol=0, atol=1e-7)


def test_param_groups_split_by_dimension() -> None:
    net = small_net()
    groups = optim.param_groups(net, 0.05)
    assert [g["weight_decay"] for g in groups] == [0.05, 0.0]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in net.parameters()), "every parameter in exactly one group"
    assert all(p.ndim > 1 for p in groups[0]["params"]) and all(p.ndim <= 1 for p in groups[1]["params"])


def test_param_groups_skip_frozen() -> None:
    net = small_net()
    for p in net.features.parameters():
        p.requires_grad = False
    n = sum(len(g["params"]) for g in optim.param_groups(net, 0.05))
    assert n == 2, "only the head's weight and bias are trainable"


def test_augment_safe_flips_and_shifts_only() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.arange(2 * 1 * 6 * 6, dtype=torch.float32).view(2, 1, 6, 6)
    out = augment.augment_safe(x, g, max_shift=0)
    for i in range(2):
        assert torch.equal(out[i], x[i]) or torch.equal(out[i], x[i].flip(-1)), "with max_shift 0: mirror or nothing"
    x = torch.arange(32 * 36, dtype=torch.float32).view(32, 1, 6, 6)
    out = augment.augment_safe(x, torch.Generator().manual_seed(1), max_shift=2, fill=-7.0)
    assert out.shape == x.shape
    assert set(out[out != -7.0].tolist()) <= set(x.flatten().tolist()), \
        "values must come from the image or the fill: no torch.roll, no interpolation"
    assert (out == -7.0).any(), "a shift brings in padding with the fill value"
    assert set(out[out != -7.0].tolist()) != set(x.flatten().tolist()), "pixels shifted out of the frame are gone"


def test_augment_safe_is_reproducible_and_leaves_input_alone() -> None:
    x = torch.randn(5, 1, 16, 16)
    before = x.clone()
    a = augment.augment_safe(x, torch.Generator().manual_seed(4), max_shift=3)
    b = augment.augment_safe(x, torch.Generator().manual_seed(4), max_shift=3)
    assert torch.equal(a, b), "the same generator seed must give the same batch"
    assert torch.equal(x, before), "do not modify the input in place"


@pytest.mark.parametrize("name", ["none", "safe", "jitter", "blur"])
def test_named_augmentations_keep_shape(name: str) -> None:
    f = augment.get_augment(name)
    x = torch.randn(3, 1, 32, 32)
    out = x if f is None else f(x, torch.Generator().manual_seed(0), fill=0.0)
    assert out.shape == x.shape
