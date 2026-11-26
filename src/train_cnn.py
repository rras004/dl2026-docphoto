"""Train a model from a config, log the run, save the checkpoint. Given.

    python -m src.train_cnn --config configs/lab4_small_cnn.yaml

Lab 3's loop, grown up. The order inside one step is the lecture's starting
configuration:

    zero_grad -> forward -> loss -> backward -> clip the gradient norm -> step -> scheduler step

and it adds what a convolutional network on images needs that an MLP on a
pixel table did not:

* **inputs moved to the device one batch at a time**, so the whole table
  stays in ordinary memory;
* **batched prediction**. Lab 3's `validate` predicts 1024 rows at once; at
  256 px that would need about 4 GB of activations for the first layer alone;
* **augmentation** on the training batches, never on validation (`src/data/augment.py`);
* **gradient clipping** and the gradient norm logged before clipping: a norm
  that explodes is the earliest sign of a learning rate that is too high;
* **mixed precision** (`amp: bf16`) and **gradient accumulation**
  (`accum_steps`), the second and third rungs of the out-of-memory ladder,
  both on a GPU only;
* **the last or the best epoch** (`select`). Choosing the best epoch on the
  validation set and then reporting that validation score is peeking: the
  number is optimistic. It is here so you can see by how much.

Every run is a W&B run: the full config and the reproducibility fingerprint,
per step the learning rate and the loss, per epoch the train and validation
RMSE, the gradient norm and the seconds, and at the end the validation
predictions as a table and the regression report as an image. The checkpoint
goes up as an artifact when `train.log_checkpoint` is true; it is off by
default so that an ablation of ten runs does not upload ten checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import time
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src import config, evaluate, tracking
from src.data.augment import get_augment
from src.data.dataset import load_table
from src.data.inputs import load_inputs, standardise_inputs
from src.data.splits import build_splits
from src.losses import make_loss
from src.models.build import build_model
from src.optim import make_adamw, make_schedule
from src.seed import set_seed


def fingerprint_or_note() -> dict:
    """Lab 5's reproducibility fingerprint, once it exists; until then a note saying so."""
    try:
        from src.repro import fingerprint
        return fingerprint()
    except (ImportError, NotImplementedError) as e:
        return {"missing": f"{type(e).__name__}: src.repro.fingerprint (lab 5, TASK 4)"}


def amp_context(device: str, amp: str):
    """bf16 autocast on a GPU when asked for; full precision everywhere else."""
    if amp == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@dataclass
class Batches:
    """Every input row on the CPU, handed out one batch at a time.

    uint8 images (grey pixels, lab 5's colour images) stay bytes until a
    batch is needed and are then scaled to [0, 1]. Grey pixels and float
    tables (lab 5's embeddings) are standardised with one mean and one scale
    fitted on the training rows only, as in lab 3; colour images for a
    pretrained network are not, because the model applies ImageNet's
    statistics itself.
    """
    X: torch.Tensor
    scaler: dict | None
    fill: float          # the padding grey in the units the model receives

    def get(self, idx, device: str) -> torch.Tensor:
        return self.prepare(self.X[torch.as_tensor(idx)], device)

    def iterate(self, batch_rows: list, device: str):
        """The batches for a list of row arrays, in order. Lab 6's photo table loads them in
        DataLoader worker processes while the model works on the previous one."""
        if hasattr(self.X, "loader"):
            for xb in self.X.loader(batch_rows, pin_memory=device == "cuda"):
                yield self.prepare(xb, device)
        else:
            for rows in batch_rows:
                yield self.get(rows, device)

    def prepare(self, xb: torch.Tensor, device: str) -> torch.Tensor:
        xb = xb.to(device, non_blocking=True)
        if xb.dtype == torch.uint8:
            xb = xb.float().div_(255)
        elif xb.dtype == torch.float16:          # lab 5's cached trunk maps
            xb = xb.float()
        if self.scaler is not None:
            xb = (xb - self.scaler["mean"]) / self.scaler["scale"]
        return xb


def fit_scaler(X: np.ndarray, train_idx: np.ndarray, chunk: int = 256) -> dict:
    """Mean and standard deviation over all values of the training rows (in [0, 1] for bytes).

    Lab 3's Standardiser, computed in chunks so that a large uint8 table is
    never turned into floats all at once.
    """
    s = s2 = n = 0.0
    for i in range(0, len(train_idx), chunk):
        x = X[train_idx[i:i + chunk]].astype(np.float64)
        if X.dtype == np.uint8:
            x /= 255.0
        s, s2, n = s + x.sum(), s2 + (x ** 2).sum(), n + x.size
    mean = s / n
    return {"mean": float(mean), "scale": float(np.sqrt(max(s2 / n - mean ** 2, 0.0))) + 1e-8}


def make_batches(X: np.ndarray, train_idx: np.ndarray, standardise: bool = True) -> Batches:
    if hasattr(X, "loader"):                    # lab 6: photographs read from disk batch by batch
        return Batches(X, None, 0.5)
    scaler = fit_scaler(X, train_idx) if standardise else None
    fill = (0.5 - scaler["mean"]) / scaler["scale"] if scaler else 0.5
    return Batches(torch.from_numpy(np.ascontiguousarray(X)), scaler, fill)


@torch.no_grad()
def predict(model, batches: Batches, idx, loss, device: str, batch_size: int = 64, amp: str = "none") -> np.ndarray:
    """Predictions in [0, 1] for the rows `idx`, in eval mode, `batch_size` at a time."""
    model.eval()
    out = []
    for xb in batches.iterate([idx[s:s + batch_size] for s in range(0, len(idx), batch_size)], device):
        with amp_context(device, amp):
            o = model(xb)
        out.append(loss.predict(o.float()).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def raw_outputs(model, batches: Batches, idx, device: str, batch_size: int = 64) -> torch.Tensor:
    """The model's outputs for the rows `idx` before the loss turns them into predictions (logits)."""
    model.eval()
    return torch.cat([model(xb).float().cpu()
                      for xb in batches.iterate([idx[s:s + batch_size] for s in range(0, len(idx), batch_size)], device)])


def n_batches(n_rows: int, batch_size: int) -> int:
    """Batches per epoch. A last batch of one row is dropped: BatchNorm cannot normalise a single image."""
    return n_rows // batch_size + (1 if n_rows % batch_size >= 2 else 0)


def train_epoch(model, optimizer, scheduler, loss, batches: Batches, idx: np.ndarray, y: torch.Tensor,
                batch_size: int, generator: torch.Generator, device: str, augment=None, aug_generator=None,
                clip_norm: float | None = 1.0, accum_steps: int = 1, amp: str = "none", on_step=None) -> dict:
    """One pass over the rows `idx` in shuffled minibatches. Returns the epoch's statistics.

    `train_rmse` here is computed from the outputs of the training steps
    themselves, in training mode, on augmented inputs and with dropout on:
    free, and good enough to see over- and underfitting. Lab 3 ran a second
    pass in eval mode, which at 256 px would double the cost of an epoch.

    `data_wait` is the share of the epoch spent waiting for the next batch.
    Near zero, the model is the bottleneck; large, the GPU waits for the
    disk and the decoder (more `num_workers`, lab 6).
    """
    model.train()
    order = idx[torch.randperm(len(idx), generator=generator).numpy()]
    nb = n_batches(len(order), batch_size)
    batch_rows = [order[b * batch_size:(b + 1) * batch_size] for b in range(nb)]
    total = se = wait = 0.0
    seen = 0
    norms = []
    start = time.time()
    stream = batches.iterate(batch_rows, device)
    for b, rows in enumerate(batch_rows):
        t0 = time.time()
        xb = next(stream)
        wait += time.time() - t0
        yb = y[torch.as_tensor(rows)].to(device)
        if augment is not None:
            xb = augment(xb, aug_generator, fill=batches.fill)
        if b % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp):
            out = model(xb)
        value = loss(out.float(), yb)
        (value / accum_steps).backward()
        if (b + 1) % accum_steps == 0 or b == nb - 1:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm or float("inf"))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            norms.append(float(norm))
            if on_step is not None:
                on_step(optimizer.param_groups[0]["lr"], value.item())
        total += value.item() * len(rows)
        se += float(((loss.predict(out.detach().float()) - yb) ** 2).sum())
        seen += len(rows)
    return {"train_loss": total / seen, "train_rmse": math.sqrt(se / seen),
            "grad_norm": float(np.mean(norms)), "lr": optimizer.param_groups[0]["lr"],
            "data_wait": wait / max(time.time() - start, 1e-9)}


def train_model(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, *,
                device: str | None = None, tags: list[str] | None = None, name: str | None = None,
                group: str | None = None, job_type: str | None = None, y: np.ndarray | None = None,
                save: bool = True, log=print) -> dict:
    """Train on the rows `train_idx` of `X`, validate on `val_idx` after every epoch, log everything.

    `X` has one row per row of `df`. `y` replaces the training targets (the
    shuffled-label check); validation is always scored against the real
    label, on the scoreable rows. Returns the history, the validation
    predictions of the selected epoch, the model, the loss, the checkpoint
    path, the W&B page, the learning rate of every step and the report table.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = name or cfg["name"]
    set_seed(cfg["seed"])
    tr = cfg["train"]
    amp, accum, bs = tr.get("amp", "none"), int(tr.get("accum_steps", 1)), int(tr["batch_size"])

    y_train_all = (df[config.TARGET].to_numpy() if y is None else np.asarray(y)).astype(np.float32)
    y_t = torch.from_numpy(y_train_all)
    y_val = df[config.TARGET].to_numpy()[val_idx]
    ok = df["eval_ok"].to_numpy()[val_idx] == 1

    batches = make_batches(X, train_idx, standardise_inputs(cfg))
    loss = make_loss(cfg, y_train_all[train_idx])
    model = build_model(cfg, tuple(X.shape[1:]), loss).to(device)
    if device == "cuda" and X.ndim == 4:
        model = model.to(memory_format=torch.channels_last)
    base = model                                  # what gets saved: the module itself, never its compiled wrapper
    if tr.get("compile", False) and device == "cuda":
        model = torch.compile(model)
    optimizer = make_adamw(model, cfg)
    steps_per_epoch = math.ceil(n_batches(len(train_idx), bs) / accum)
    scheduler = make_schedule(optimizer, tr["epochs"] * steps_per_epoch, tr)
    augment = get_augment(tr.get("augment", "none"))
    generator = torch.Generator().manual_seed(cfg["seed"])
    aug_generator = torch.Generator().manual_seed(cfg["seed"] + 1)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logged = {**cfg, "name": name, "fingerprint": fingerprint_or_note(), "device": device,
              "n_train_rows": int(len(train_idx)), "n_val_rows": int(len(val_idx)),
              "parameters": n_params, "trainable_parameters": n_train}
    tracking.init_run(logged, tags=tags, name=name, group=group, job_type=job_type)
    log(f"{name}: {len(train_idx)} train / {len(val_idx)} val rows, input {tuple(X.shape[1:])}, "
        f"{n_params:,} parameters ({n_train:,} trainable), {device}"
        + (f", accumulating {accum} batches of {bs}" if accum > 1 else ""))

    history, lr_steps = [], []
    step = 0

    def on_step(lr, value):
        nonlocal step
        step += 1
        lr_steps.append(lr)
        if step % 10 == 0:
            tracking.log_metrics({"step/lr": lr, "step/loss": value}, step=step)

    best = (float("inf"), None, None, 0)
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, tr["epochs"] + 1):
            t0 = time.time()
            stats = train_epoch(model, optimizer, scheduler, loss, batches, train_idx, y_t, bs, generator, device,
                                augment, aug_generator, tr.get("clip_norm", 1.0), accum, amp, on_step)
            pred = predict(model, batches, val_idx, loss, device, amp=amp)
            row = {"epoch": epoch, **stats, "val_rmse": evaluate.rmse(y_val[ok], pred[ok]),
                   "val_mae": evaluate.mae(y_val[ok], pred[ok]), "seconds": time.time() - t0}
            if device == "cuda":
                row["max_memory_MB"] = torch.cuda.max_memory_allocated() / 1e6
            history.append(row)
            tracking.log_metrics({f"epoch/{k}": v for k, v in row.items()}, step=max(step, 1))
            if row["val_rmse"] < best[0]:
                best = (row["val_rmse"], copy.deepcopy({k: v.detach().cpu() for k, v in base.state_dict().items()}),
                        pred, epoch)
            log(f"epoch {epoch:3d}  train loss {stats['train_loss']:.4f}  train RMSE {stats['train_rmse']:.4f}  "
                f"val RMSE {row['val_rmse']:.4f}  grad norm {stats['grad_norm']:.3f}  lr {stats['lr']:.2e}  "
                f"{row['seconds']:.0f} s")

        selected = len(history)
        if tr.get("select", "last") == "best":
            base.load_state_dict(best[1])
            pred, selected = best[2], best[3]
            log(f"keeping epoch {selected} of {len(history)}: chosen on the validation set, so the "
                f"validation score below is optimistic (peeking)")
        score = evaluate.rmse(y_val[ok], pred[ok])

        ckpt = None
        if save:
            ckpt = config.ROOT / "models" / f"{name}.pt"
            ckpt.parent.mkdir(exist_ok=True)
            torch.save({"state_dict": base.state_dict(), "config": cfg, "in_shape": tuple(X.shape[1:]),
                        "loss_state": loss.state(), "scaler": batches.scaler, "fingerprint": logged["fingerprint"],
                        "epoch": selected, "val_rmse": score}, ckpt)

        vdf = df.iloc[val_idx]
        table, fig = evaluate.regression_report(vdf[ok], y_val[ok], pred[ok], baseline=float(loss.prior),
                                                title=name)
        tracking.log_figure("val/regression_report", fig)
        plt.close(fig)
        tracking.log_table("val/predictions", pd.DataFrame({
            "image_id": vdf["image_id"].to_numpy()[ok], "true": y_val[ok], "pred": pred[ok],
            "residual": pred[ok] - y_val[ok], "family": vdf["family"].to_numpy()[ok],
            "photographer": vdf["photographer"].to_numpy()[ok]}))
        if loss.n_out > 1:      # a classification loss (lab 5's ce_bins): the confusion matrix of its bins
            classes = raw_outputs(model, batches, val_idx[ok], device).argmax(1).numpy()
            truth = np.clip((y_val[ok] * loss.n_out).astype(int), 0, loss.n_out - 1)
            tracking.log_confusion("val/confusion_bins", truth, classes,
                                   [f"{k / loss.n_out:.1f}-{(k + 1) / loss.n_out:.1f}" for k in range(loss.n_out)])
        tracking.log_metrics({"final/val_rmse": score, "final/val_mae": table.loc["model", "mae"],
                              "final/spread_ratio": table.loc["model", "spread ratio"],
                              "final/bias": table.loc["model", "bias"], "final/epoch": selected})
        if ckpt is not None and tr.get("log_checkpoint", False):
            tracking.log_checkpoint(ckpt, name, {"val_rmse": score, "epoch": selected})
        url = tracking.run_url()
    finally:
        tracking.finish()
    return {"history": pd.DataFrame(history), "val_pred": pred, "model": model, "loss": loss,
            "checkpoint": ckpt, "url": url, "val_index": val_idx, "ok": ok, "lr_steps": lr_steps,
            "report": table, "selected_epoch": selected, "scaler": batches.scaler}


def lr_range_test(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, *, lr_min: float = 1e-6,
                  lr_max: float = 1.0, steps: int = 100, device: str | None = None) -> tuple[pd.DataFrame, float]:
    """Raise the learning rate exponentially from lr_min to lr_max, one batch per step, and record the loss.

    The loss falls once the rate is large enough to learn and explodes once it
    is too large. A good starting rate is about ten times smaller than the
    one at the bottom of the smoothed curve. Returns the curve and that
    suggestion. One W&B run (job_type lr_range_test).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]
    y_t = torch.from_numpy(df[config.TARGET].to_numpy().astype(np.float32))
    batches = make_batches(X, train_idx, standardise_inputs(cfg))
    loss = make_loss(cfg, y_t.numpy()[train_idx])
    model = build_model(cfg, tuple(X.shape[1:]), loss).to(device)
    optimizer = make_adamw(model, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    tracking.init_run({**cfg, "lr_min": lr_min, "lr_max": lr_max}, name=f"{cfg['name']}_lr_range",
                      job_type="lr_range_test")
    rows, smooth, best = [], 0.0, float("inf")
    order = np.zeros(0, dtype=int)
    model.train()
    try:
        for i in range(steps):
            lr = lr_min * (lr_max / lr_min) ** (i / (steps - 1))
            for g in optimizer.param_groups:
                g["lr"] = lr
            if len(order) < tr["batch_size"]:
                order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
            rows_b, order = order[:tr["batch_size"]], order[tr["batch_size"]:]
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(batches.get(rows_b, device)), y_t[torch.as_tensor(rows_b)].to(device))
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("clip_norm", 1.0) or float("inf"))
            optimizer.step()
            smooth = 0.9 * smooth + 0.1 * value.item()
            s = smooth / (1 - 0.9 ** (i + 1))
            rows.append({"lr": lr, "loss": value.item(), "smoothed": s})
            tracking.log_metrics({"lr": lr, "loss": value.item(), "smoothed": s}, step=i)
            best = min(best, s)
            if not math.isfinite(s) or s > 4 * best:
                break
    finally:
        tracking.finish()
    curve = pd.DataFrame(rows)
    return curve, float(curve.loc[curve["smoothed"].idxmin(), "lr"] / 10)


def main() -> None:
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="Train a model from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = load_inputs(cfg, df)
    sp = cfg["split"]
    train_idx, val_idx = build_splits(df, sp["group_column"], sp["val_fraction"], sp["seed"])
    result = train_model(cfg, X, df, train_idx, val_idx)
    print(f"val RMSE {evaluate.rmse(df[config.TARGET].to_numpy()[val_idx][result['ok']], result['val_pred'][result['ok']]):.4f}"
          f"  checkpoint {result['checkpoint']}  run {result['url']}")


if __name__ == "__main__":
    main()
