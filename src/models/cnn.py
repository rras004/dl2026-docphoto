"""Convolutional networks, trained from scratch.

Lab 3's MLP had one weight per pixel per unit. A detector it learned at one
position did not exist at any other, and at 64 px there were no strokes to
detect anyway. A convolution answers both. Its filter is a few pixels wide
and slides over the whole image, so one detector serves every position
(weight sharing), and it costs so little per pixel that it can run at four
times the resolution.

Given here: `conv_block`, `init_weights`, `prior_bias`, `ResNetSmall`.
Yours: `SmallCNN` (TASK 1), and if you are fast, `ResBlock` (TASK ⭐).

**Models return logits** from lab 4 on: the last layer is a plain Linear and
`src/losses.py` applies the sigmoid. See the docstring there for why.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_norm(kind: str, channels: int) -> nn.Module:
    """BatchNorm, GroupNorm or nothing.

    BatchNorm normalises each channel with the statistics of the current
    batch in training and with running averages in evaluation, which is why
    `model.eval()` before prediction is not optional. At batch sizes of a few
    images (a ResNet at 768 px on a laptop GPU), the batch statistics are
    noise; GroupNorm normalises within each image and does not care about
    the batch size.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


def conv_block(c_in: int, c_out: int, stride: int = 1, kernel: int = 3, norm: str = "batch") -> nn.Sequential:
    """Convolution, normalisation, ReLU. The convolution has no bias: the normalisation that
    follows would subtract it again."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        make_norm(norm, c_out),
        nn.ReLU(inplace=True),
    )


def init_weights(model: nn.Module) -> None:
    """He initialisation for convolutions followed by ReLU, unit scale for norms.

    He (Kaiming) initialisation scales each filter so that the variance of the
    activations stays roughly constant through a stack of ReLU layers; with
    the defaults it shrinks or grows layer by layer, and a deep stack starts
    either dead or saturated. A convolution marked `zero_init = True` (the
    last layer of a residual branch) starts at zero instead: the block then
    begins as the identity and the network as a shallower one.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if getattr(m, "zero_init", False):
                nn.init.zeros_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)


def prior_bias(head: nn.Linear, prior: float) -> None:
    """Small output weights and the bias at logit(prior): step 0 predicts the training mean.

    The same trick as lab 3's MLP. The target sits mostly near 1; a head that
    starts at logit 0, i.e. at 0.5, spends its first epochs moving one number.
    """
    p = min(max(float(prior), 1e-3), 1 - 1e-3)
    nn.init.normal_(head.weight, std=1e-3)
    nn.init.constant_(head.bias, math.log(p / (1 - p)))


class GlobalPool(nn.Module):
    """Given. One vector per image from a feature map: the average over all positions, the
    maximum, or both side by side (`factor` tells the head how wide the vector is).

    The average answers "how sharp is this page overall"; the maximum answers
    "how strong is the strongest response anywhere", and a defect that hits
    one column of the page is a local event. Global pooling of either kind is
    what lets the network accept an input of any size.
    """

    def __init__(self, kind: str = "avg"):
        super().__init__()
        if kind not in ("avg", "max", "avgmax"):
            raise ValueError(f"unknown pool {kind!r}")
        self.kind = kind
        self.factor = 2 if kind == "avgmax" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "avg":
            return x.mean((2, 3))
        if self.kind == "max":
            return x.amax((2, 3))
        return torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)


class SmallCNN(nn.Module):
    """TASK 1. A small convolutional network that returns one logit per image.

    Build it in this order (the names matter: the training loop and the
    checks use `self.features` and `self.head`):

    1. `self.features`, an nn.Sequential of
       * the stem: `conv_block(in_ch, widths[0], stride=stem_stride, kernel=stem_kernel, norm=norm)`.
         It reads every input pixel and reduces the resolution at once. Every
         later layer then works on a map stem_stride² times smaller: what the
         stem's filters saw at full resolution survives in their responses,
         and on a CPU this is the difference between one and many minutes;
       * for each further width w (with the previous width as input):
         `conv_block(previous, w, norm=norm)` followed by `nn.MaxPool2d(2)`;
    2. `self.pool = GlobalPool(pool)`: one vector per image;
    3. `self.drop = nn.Dropout(dropout)`;
    4. `self.head = nn.Linear(widths[-1] * self.pool.factor, 1)`;
    5. `init_weights(self)`, then `prior_bias(self.head, prior)`, in this order.

    `forward(x)` takes (batch, in_ch, H, W) and returns shape (batch,):
    features, pool, dropout, head, `squeeze(1)`. No sigmoid.

    For the first in-class run (configs/lab4_small_cnn.yaml: in_ch 1,
    widths [8, 16, 32, 64], stem_stride 2, stem_kernel 5, pool "avgmax",
    norm "batch", dropout 0.0) and a 1 x 256 x 256 input this is exactly:

        stem     conv_block(1, 8, stride=2, kernel=5)          ->  8 x 128 x 128
        block 1  conv_block(8, 16),  MaxPool2d(2)              -> 16 x  64 x  64
        block 2  conv_block(16, 32), MaxPool2d(2)              -> 32 x  32 x  32
        block 3  conv_block(32, 64), MaxPool2d(2)              -> 64 x  16 x  16
        pool     GlobalPool("avgmax"): mean and max per channel -> 128
        drop     nn.Dropout(0.0)                                -> 128
        head     nn.Linear(128, 1)                              -> 1 logit

    24,761 parameters. The second run (768 px, stem_stride 4, stem_kernel 7)
    builds the same layers with a 7 x 7 stride-4 stem: 24,953 parameters.
    """

    def __init__(self, in_ch: int = 1, widths: list[int] = (16, 32, 64, 128), norm: str = "batch",
                 dropout: float = 0.0, prior: float = 0.5, stem_stride: int = 2, stem_kernel: int = 5,
                 pool: str = "avg"):
        super().__init__()
        raise NotImplementedError("TASK 1: build the layers of SmallCNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK 1: the forward pass of SmallCNN")


class ResBlock(nn.Module):
    """TASK ⭐ (for the fast). A pre-norm residual block that keeps the shape of its input.

    out = x + conv2(relu(norm2(conv1(relu(norm1(x))))))

    with 3 x 3 convolutions of `channels` to `channels`, padding 1, no bias,
    and norms from `make_norm(norm, channels)`. "Pre-norm": normalisation and
    activation come *before* each convolution, so the identity path from
    input to output carries nothing but additions, which is what lets
    gradients reach the first layers of a deep stack unchanged.

    Mark the last convolution with `self.conv2.zero_init = True`;
    `init_weights` then starts it at zero and the block as the identity.
    """

    def __init__(self, channels: int, norm: str = "batch"):
        super().__init__()
        raise NotImplementedError("TASK ⭐: the layers of ResBlock")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK ⭐: the forward pass of ResBlock")


class ResNetSmall(nn.Module):
    """Given: the SmallCNN layout with a residual block in every stage. Needs your ResBlock.

    stem -> ResBlock -> for each further width: a strided conv_block that
    halves the resolution and changes the width, then a ResBlock -> norm,
    ReLU -> pool -> dropout -> head. The final norm and ReLU are there
    because a pre-norm block leaves its output unnormalised.
    """

    def __init__(self, in_ch: int = 1, widths: list[int] = (16, 32, 64, 128), norm: str = "batch",
                 dropout: float = 0.0, prior: float = 0.5, stem_stride: int = 2, stem_kernel: int = 5,
                 pool: str = "avg"):
        super().__init__()
        layers: list[nn.Module] = [conv_block(in_ch, widths[0], stride=stem_stride, kernel=stem_kernel, norm=norm),
                                   ResBlock(widths[0], norm)]
        for c_in, c_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [conv_block(c_in, c_out, stride=2, norm=norm), ResBlock(c_out, norm)]
        layers += [make_norm(norm, widths[-1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = GlobalPool(pool)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(widths[-1] * self.pool.factor, 1)
        init_weights(self)
        prior_bias(self.head, prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.pool(self.features(x)))).squeeze(1)
