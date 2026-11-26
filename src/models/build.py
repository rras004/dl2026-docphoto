"""One function that turns a config into a model. Given.

The training loop, `predict.py` and every later lab build models through
`build_model`, so that a checkpoint (which stores its config) can always be
turned back into the network that produced it:

    loss = losses.from_state(ckpt["loss_state"])
    model = build_model(ckpt["config"], ckpt["in_shape"], loss)
    model.load_state_dict(ckpt["state_dict"])

`in_shape` is the shape of one input without the batch dimension, e.g.
(1, 256, 256). The loss decides the number of outputs and where the output
layer's bias starts.
"""
from __future__ import annotations

import torch.nn as nn

from src.losses import Loss
from src.models import cnn


def build_model(cfg: dict, in_shape: tuple, loss: Loss) -> nn.Module:
    m = cfg["model"]
    kind = m["type"]
    if loss.n_out != 1:
        raise ValueError(f"{kind} has one output; the loss {loss.name!r} needs {loss.n_out}")
    common = dict(widths=list(m["widths"]), norm=m.get("norm", "batch"), dropout=m.get("dropout", 0.0),
                  stem_stride=m.get("stem_stride", 2), stem_kernel=m.get("stem_kernel", 5), pool=m.get("pool", "avg"),
                  prior=loss.prior)
    if kind == "small_cnn":
        model = cnn.SmallCNN(in_ch=in_shape[0], **common)
    elif kind == "resnet_small":
        model = cnn.ResNetSmall(in_ch=in_shape[0], **common)
    else:
        raise ValueError(f"unknown model type {kind!r}")
    loss.init_head(model.head)
    return model
