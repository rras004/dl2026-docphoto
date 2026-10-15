"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int],
                 dropout: float = 0.3, prior: float = 0.5):
        super().__init__()

        layers = []
        current_dim = in_dim

        for h in hidden:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h

        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

        p = min(max(prior, 1e-6), 1 - 1e-6)

        with torch.no_grad():
            self.network[-1].bias.fill_(
                torch.logit(torch.tensor(p)).item()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        output = torch.sigmoid(output)
        return output.squeeze(-1)
