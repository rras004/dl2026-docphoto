"""Checks to run before you tune anything.

A training run that "does not learn" has a hundred possible causes and a
learning curve that looks the same for all of them. These checks each take
seconds and each rules out a family of causes:

1. **shape and range** of one forward pass (`check_output`): the model maps
   a batch to one number per image, finite, and the loss turns that number
   into a prediction in [0, 1];
2. **the initial loss** (`initial_loss`): before any training, the loss
   should equal what the training-mean predictor scores, because the output
   bias starts there. For classification over K classes the textbook value
   is ln K; for this regression it is the variance of the targets. Far
   above that = the initialisation is wrong;
3. **overfit one batch** (`overfit_one_batch`, TASK 2): a model that cannot
   drive the loss on sixteen photographs to nearly zero has a bug (wrong
   target, detached graph, frozen weights, a learning rate of zero) and no
   amount of data will help it;
4. **the same seed twice** (`same_seed_twice`, TASK 2): two runs with the
   same seed must give the same losses, or no comparison between two runs
   means anything;
5. **shuffled labels**: trained on permuted targets, the model can memorise
   the training set but must stay at the constant predictor on validation.
   If it beats the constant, information leaks from somewhere other than
   the image. That one is a full run; the notebook does it.

Plus the memory budget, which you should estimate before you run and then
measure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.seed import set_seed  # noqa: F401  (TASK 2b needs it)


@torch.no_grad()
def check_output(model: nn.Module, xb: torch.Tensor, loss) -> torch.Tensor:
    """Run one batch in eval mode; assert the shape (batch,), finite values, predictions in [0, 1]."""
    model.eval()
    out = model(xb)
    assert out.shape == (len(xb),), f"expected shape ({len(xb)},), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite output"
    p = loss.predict(out)
    assert float(p.min()) >= 0 and float(p.max()) <= 1, "predictions outside [0, 1]"
    return p


@torch.no_grad()
def initial_loss(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss) -> tuple[float, float]:
    """(loss of the untrained model, loss of predicting the training mean) on one batch, eval mode."""
    model.eval()
    untrained = float(loss(model(xb), yb))
    constant = float(((yb - loss.prior) ** 2).mean())
    return untrained, constant


def overfit_one_batch(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss,
                      steps: int = 100, lr: float = 1e-3) -> list[float]:
    """TASK 2a. Train on this one batch for `steps` steps; return the loss of every step.

    Training mode, `torch.optim.AdamW(model.parameters(), lr=lr)` with no
    weight decay, and per step: zero_grad, forward, loss, backward, step.
    Record `loss.item()` before the step. The list should fall to nearly
    zero; with sixteen photographs and a hundred steps, under 1e-3.
    """
    raise NotImplementedError("TASK 2a: overfit one batch")


def same_seed_twice(make_model, xb: torch.Tensor, yb: torch.Tensor, loss, seed: int = 0,
                    steps: int = 5) -> tuple[list[float], list[float]]:
    """TASK 2b. Two short runs from the same seed; return both lists of losses.

    `make_model()` builds a fresh model. For each of the two runs: `set_seed(seed)`,
    build the model, and train it with `overfit_one_batch` for `steps` steps.
    The caller compares the two lists.
    """
    raise NotImplementedError("TASK 2b: the same seed, twice")


def memory_estimate(model: nn.Module, xb: torch.Tensor) -> dict:
    """Estimate the training memory of one step at this batch size, before running it.

    * weights, gradients and AdamW's two moment estimates, all in float32:
      4 numbers of 4 bytes per parameter = **16 bytes per parameter**;
    * activations: every layer's output is kept for the backward pass.
      Measured here with forward hooks on one batch. They grow linearly with
      the batch size and with the number of pixels, which is why they, not
      the parameters, decide whether a convolutional network fits.

    An estimate: in-place ReLUs share memory with their input, and the
    framework keeps a few extra buffers. Measure on a GPU with
    `torch.cuda.max_memory_allocated()` (`measure_step_memory`).
    """
    n_params = sum(p.numel() for p in model.parameters())
    sizes: list[int] = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: sizes.append(out.numel() * out.element_size()))
             for m in model.modules() if len(list(m.children())) == 0]
    model.train()
    with torch.no_grad():
        model(xb)
    for h in hooks:
        h.remove()
    act = float(sum(sizes))
    return {"parameters": n_params, "params+grads+adamw MB": 16 * n_params / 1e6,
            "activations MB": act / 1e6, "batch": len(xb), "activations per image MB": act / len(xb) / 1e6,
            "input batch MB": xb.numel() * xb.element_size() / 1e6}


def measure_step_memory(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss) -> float | None:
    """Peak GPU memory of one training step in MB, or None on a CPU."""
    if not torch.cuda.is_available() or xb.device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    opt.zero_grad(set_to_none=True)
    loss(model(xb), yb).backward()
    opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def spread(values) -> float:
    """Largest absolute difference between two equal-length lists."""
    a, b = (np.asarray(v, dtype=float) for v in values)
    return float(np.max(np.abs(a - b)))


def count_macs(model: nn.Module, xb: torch.Tensor) -> dict:
    """Multiply-accumulates of one forward pass per image, by layer type, counted with hooks.

    A convolution with c_in inputs, c_out outputs and a k x k kernel costs
    c_in * c_out * k * k multiply-adds at every position of its output map.
    The map size is therefore what decides the cost: halve the resolution
    after the stem and every later layer costs a quarter. A training step
    costs about three forward passes (forward, and twice that for backward).
    """
    total = {"conv": 0, "linear": 0}

    def conv_hook(m, inp, out):
        total["conv"] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        total["linear"] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(xb[:1])
    for h in hooks:
        h.remove()
    return {"MMAC per image": (total["conv"] + total["linear"]) / 1e6,
            "conv share": total["conv"] / max(1, total["conv"] + total["linear"])}
