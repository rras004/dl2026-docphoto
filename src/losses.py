"""The loss, and how a model output becomes a prediction.

Given in lab 4. Lab 5 replaces this file with one in which you write the
alternatives.

**The convention from lab 4 on: models return logits.** A network's last
layer is a plain Linear, and the activation that maps its output into [0, 1]
belongs to the loss and to the prediction, not to the model. Two reasons.
Losses such as binary cross-entropy are numerically stable only when they get
the logit (`BCEWithLogitsLoss` does the sigmoid and the log together); and a
model that already applies a sigmoid, fed to a loss that applies another,
trains without an error message and predicts only between 0.5 and 0.73: the
double-sigmoid bug. Lab 3's MLP still ends in a sigmoid; its output kind is
`prob`, and the loss knows not to squash it again.

A `Loss` here is a small object with four jobs:

    loss(out, y)        the number to minimise, from the raw model output
    loss.predict(out)   the usability prediction in [0, 1] from the raw output
    loss.init_head(h)   set the output layer's bias so that step 0 predicts the training mean
    loss.state()        what predict.py needs to rebuild it from the checkpoint
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def output_kind(cfg: dict) -> str:
    """'prob' for lab 3's MLP, which ends in a sigmoid; 'logit' for everything from lab 4 on."""
    return "prob" if cfg["model"]["type"] == "mlp" else "logit"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


class Loss:
    name = "base"
    n_out = 1

    def __init__(self, prior: float, output: str = "logit"):
        self.prior = float(prior)      # the training mean of the target
        self.output = output

    def __call__(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_head(self, head: nn.Linear) -> None:
        """Small weights, bias at logit(prior): the untrained model predicts the training mean."""
        nn.init.normal_(head.weight, std=1e-3)
        nn.init.constant_(head.bias, _logit(self.prior))

    def state(self) -> dict:
        return {"name": self.name, "prior": self.prior, "output": self.output}


class MSE(Loss):
    """Squared error of the usability prediction. The competition metric is its square root."""
    name = "mse"

    def __call__(self, out, y):
        return F.mse_loss(self.predict(out), y)

    def predict(self, out):
        return torch.sigmoid(out) if self.output == "logit" else out


LOSSES = {"mse": MSE}


def make_loss(cfg: dict, y_train: np.ndarray) -> Loss:
    """The loss named by cfg['train']['loss'] (default 'mse'), fitted to the training targets."""
    name = cfg["train"].get("loss", "mse")
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; this file knows {sorted(LOSSES)}")
    return LOSSES[name](prior=float(np.mean(y_train)), output=output_kind(cfg))


def from_state(state: dict) -> Loss:
    """Rebuild the loss from `checkpoint['loss_state']`, for prediction."""
    loss = LOSSES[state["name"]](prior=state["prior"], output=state["output"])
    loss.__dict__.update({k: v for k, v in state.items() if k not in ("name",)})
    return loss
