"""Data augmentation on a batch of images, training only.

Augmentation shows the network variations of each training photograph that
leave the *label* unchanged, so that it learns the label and not the
photograph. The condition is the whole point: on this task the label is how
readable the page is, so a transformation that changes readability does not
make a new training example, it makes a wrong one.

* **Safe**: a horizontal mirror (an OCR engine reads a mirrored page no
  worse or better as far as blur, light and creases go; the strokes are the
  same strokes) and a shift of a few pixels (where the page lies in the frame
  is not what the label measures). `augment_safe`, TASK 4.
* **Label-corrupting**, given so that you can measure them: brightness and
  contrast jitter (exposure is one of the causes of a low label) and blur
  (*the* cause of a low label). Both are standard on ImageNet, where the
  label is "a dog", and wrong here.

All functions take a batch `x` of shape (batch, channels, H, W) on any
device, a CPU `torch.Generator` for reproducibility, and `fill`, the value
that pixels shifted in from outside the image take (the grey of the padding,
in whatever units the input has: after standardisation it is not 0.5). They
return a new batch of the same shape and never touch the input in place.

A separate generator from the one that shuffles the minibatches: otherwise
switching augmentation on would also change the order of the batches, and a
comparison "with against without" would compare two things at once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (TASK 4 needs it)
from torchvision.transforms.v2 import functional as TF


def augment_safe(x: torch.Tensor, generator: torch.Generator, max_shift: int = 8, fill: float = 0.0) -> torch.Tensor:
    """TASK 4. Mirror each image left-right with probability 1/2, then shift it by up to `max_shift` px.

    Per image, not per batch: draw a (batch,) mask with `torch.rand(len(x), generator=generator) < 0.5`
    and a (batch, 2) integer shift with `torch.randint(-max_shift, max_shift + 1, (len(x), 2), generator=generator)`.
    Both are drawn on the CPU (the generator lives there); move the mask to `x.device`.

    * the mirror: `torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)`;
    * the shift: pad the whole batch by `max_shift` on every side with `fill`
      (`F.pad(x, (s, s, s, s), value=fill)`), then cut each image's H x W
      window back out at its own offset. A loop over the batch is fine.
      Not `torch.roll`: it would wrap the right edge of the page round to the
      left and invent a sharp edge that is not in the photograph.
    """
    raise NotImplementedError("TASK 4: flip and shift")


def augment_jitter(x: torch.Tensor, generator: torch.Generator, fill: float = 0.0) -> torch.Tensor:
    """Given. Per image, contrast x c and brightness + b, c in [0.6, 1.4], b in [-0.4, 0.4].

    The amounts are those of a common ImageNet recipe (ColorJitter 0.4),
    applied to standardised pixels, where 0.4 is 0.4 standard deviations.
    A dark photograph made brighter keeps its dark label: a wrong example.
    """
    n = len(x)
    c = (0.6 + 0.8 * torch.rand(n, generator=generator)).to(x.device).view(-1, 1, 1, 1)
    b = (0.8 * torch.rand(n, generator=generator) - 0.4).to(x.device).view(-1, 1, 1, 1)
    return x * c + b


def augment_blur(x: torch.Tensor, generator: torch.Generator, fill: float = 0.0) -> torch.Tensor:
    """Given. Gaussian blur on a random half of the batch, sigma drawn once per batch in [0.5, 2.0] px.

    At 256 px, 2 px is the height of a letter's stroke. A sharp photograph
    blurred keeps its sharp label: a wrong example, and the most direct way
    to teach the network that blur does not matter.
    """
    sigma = float(0.5 + 1.5 * torch.rand(1, generator=generator))
    k = 2 * int(3 * sigma + 0.5) + 1
    mask = (torch.rand(len(x), generator=generator) < 0.5).to(x.device).view(-1, 1, 1, 1)
    return torch.where(mask, TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma]), x)


def compose(*functions):
    def run(x, generator, fill=0.0):
        for f in functions:
            x = f(x, generator, fill=fill)
        return x
    return run


AUGMENTS = {
    "none": None,
    "safe": augment_safe,
    "jitter": compose(augment_safe, augment_jitter),    # safe + brightness and contrast jitter
    "blur": compose(augment_safe, augment_blur),        # safe + blur
}


def get_augment(name: str):
    """The augmentation named in cfg['train']['augment'], or None."""
    if name not in AUGMENTS:
        raise ValueError(f"unknown augmentation {name!r}; choose from {sorted(AUGMENTS)}")
    return AUGMENTS[name]
