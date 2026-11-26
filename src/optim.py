"""The optimiser and the learning-rate schedule of the lecture's starting configuration.

    AdamW, two parameter groups (no weight decay on biases and normalisation)
    linear warmup, then cosine decay to zero
    gradient clipping at norm 1 (in the training loop)

Yours: `param_groups` (TASK 3). Given: `make_adamw`, `make_schedule`.

Why two groups. Weight decay pulls every parameter towards zero a little at
every step, which keeps the weights of a large layer small and the function
smooth. A bias or a normalisation scale is one number per channel that sets
an offset or a gain; pulling it to zero regularises nothing and can switch a
channel off. In PyTorch they are exactly the parameters with fewer than two
dimensions.

Why warmup. At step 0 AdamW's moment estimates are built from one or two
gradients and its steps are as large as the learning rate allows in whatever
direction those gradients point. A few hundred steps of a small learning rate
let the estimates settle. Why cosine: large steps while far from a minimum,
small ones at the end, no decision about when to drop.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """TASK 3. Split the trainable parameters into two groups for AdamW.

    Return `[{"params": [...], "weight_decay": weight_decay},
             {"params": [...], "weight_decay": 0.0}]`
    where the second group holds every parameter with `p.ndim <= 1` (biases,
    BatchNorm and GroupNorm weights and biases) and the first everything else
    (convolution and linear weights). Skip parameters with
    `requires_grad == False` (a frozen backbone in lab 5). Every trainable
    parameter in exactly one group.
    """
    raise NotImplementedError("TASK 3: the two parameter groups")


def make_adamw(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW over the two groups, learning rate and weight decay from cfg['train']."""
    tr = cfg["train"]
    return torch.optim.AdamW(param_groups(model, tr["weight_decay"]), lr=tr["lr"], betas=(0.9, 0.999))


def make_schedule(optimizer: torch.optim.Optimizer, total_steps: int, train_cfg: dict):
    """The learning-rate multiplier per optimiser step.

    schedule: none            constant
    schedule: warmup_cosine   linear from ~0 to 1 over warmup_frac of the steps, then cosine to 0

    Call `scheduler.step()` once after every `optimizer.step()`.
    """
    kind = train_cfg.get("schedule", "none")
    warm = int(round(train_cfg.get("warmup_frac", 0.1) * total_steps)) if kind == "warmup_cosine" else 0

    def factor(step: int) -> float:
        if kind == "none":
            return 1.0
        if kind != "warmup_cosine":
            raise ValueError(f"unknown schedule {kind!r}")
        if step < warm:
            return (step + 1) / (warm + 1)
        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
