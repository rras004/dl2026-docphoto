"""Experiment tracking with Weights & Biases.

Print statements vanish. A run that recorded its config, its metrics per epoch
and its final score can be compared against the twenty runs that came after it,
which is the only way to tell an improvement from noise.

Setup, once:

    cp .env.example .env            # paste your key from https://wandb.ai/authorize

Docker Compose passes `WANDB_API_KEY` and `WANDB_MODE` into the container.
**No key is also fine**: runs are then logged to an anonymous account and
you get a link you can claim later. `WANDB_MODE=offline` logs to disk and
`wandb sync wandb/latest-run` uploads later; `disabled` turns it off.
If the service cannot be reached, the run falls back to offline on its own
and says so, so a network problem never kills a training run.

What to log, the lecture's six things: the commit and whether the tree was
clean, the data version and the split seed, the full config, the environment
(image tag, GPU), the curves and not only the final number, and the
checkpoint together with its evaluation. Lab 3 logged the config, the curves
and the checkpoint; lab 5's fingerprint adds the rest.

Lab 4 version: `init_run` takes a `group` and a `job_type`, so that the folds
of one cross-validation or the arms of one ablation sit together on the W&B
page; and three helpers log a figure, a table and a confusion matrix.
"""
from __future__ import annotations

import os
from pathlib import Path

import wandb

from src import config


def init_run(cfg: dict, tags: list[str] | None = None, name: str | None = None,
             group: str | None = None, job_type: str | None = None):
    """Start a run named after the config. Returns the wandb run (or None when disabled)."""
    mode = os.environ.get("WANDB_MODE", "online")
    kwargs = dict(
        project=cfg["wandb"]["project"],
        name=name or cfg["name"],
        config=cfg,
        tags=list(cfg["wandb"].get("tags", [])) + list(tags or []),
        group=group,
        job_type=job_type,
        mode=mode,
        dir=str(config.ROOT),           # wandb/ lands at the project root, which .gitignore covers
        reinit=True,
    )
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        kwargs["anonymous"] = "allow"   # no key: an anonymous run with a claimable link
    try:
        return wandb.init(**kwargs)
    except Exception as e:  # noqa: BLE001  (anything: no network, bad key, service down)
        print(f"wandb: could not start an online run ({type(e).__name__}: {e}); logging offline instead")
        kwargs.pop("anonymous", None)
        kwargs["mode"] = "offline"
        return wandb.init(**kwargs)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dictionary of scalars, typically once per epoch."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_figure(key: str, fig) -> None:
    """A matplotlib figure, stored as an image on the run."""
    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig)})


def log_table(key: str, frame) -> None:
    """A pandas DataFrame as a W&B table: sortable, filterable, plottable on the run page."""
    if wandb.run is not None:
        wandb.log({key: wandb.Table(dataframe=frame.reset_index(drop=True))})


def log_confusion(key: str, y_true, y_pred, class_names: list[str]) -> None:
    """A confusion matrix from integer class labels, drawn by W&B."""
    if wandb.run is not None:
        wandb.log({key: wandb.plot.confusion_matrix(y_true=list(map(int, y_true)), preds=list(map(int, y_pred)),
                                                    class_names=class_names)})


def log_checkpoint(path: Path, name: str, metadata: dict | None = None) -> None:
    """Attach a checkpoint to the run as an artifact, so weights and numbers stay together."""
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name, type="model", metadata=metadata or {})
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def run_url() -> str | None:
    """The page of the current run, or None offline."""
    if wandb.run is None:
        return None
    return wandb.run.get_url() if wandb.run.settings.mode == "online" else f"(offline: {wandb.run.dir})"


def finish() -> None:
    """Close the run. Do this even when training crashes, or the run stays marked as running forever."""
    if wandb.run is not None:
        wandb.finish()
