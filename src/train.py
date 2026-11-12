"""Train the MLP on raw pixels and log the run to Weights & Biases.

    python -m src.train --config configs/lab3_mlp.yaml

Given: pixel standardisation, the epoch loop with its logging, and the
checkpoint. Yours, each a few lines, each with its contract in the docstring:

    make_optimizer   TASK 3
    train_epoch      TASK 4   the minibatch loop
    predict          TASK 5   model outputs as numpy
    validate         TASK 6   RMSE on the scoreable validation rows

Read the given parts once. There is no framework underneath: this file is the
whole of supervised training.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, tracking
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.evaluate import rmse  # noqa: F401  (TASK 6 needs it)
from src.models.mlp import MLP
from src.seed import set_seed


class Standardiser:
    """Zero mean and unit variance over all pixels, fitted on the training rows only.

    One mean and one scale for the whole image, not one per pixel: with a
    thousand training rows a per-pixel estimate is mostly noise. Fitting on
    all rows would leak the validation distribution into training; a small
    leak, but the habit matters. The two numbers go into the checkpoint so
    that prediction uses exactly the same transform.
    """

    def fit(self, X: np.ndarray) -> Standardiser:
        self.mean = float(X.mean())
        self.scale = float(X.std()) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


# --------------------------------------------------------------- your part

def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )


def train_epoch(model, optimizer, loss_fn, X, y, batch_size, generator):
    model.train()

    order = torch.randperm(len(X), generator=generator)
    total_loss = 0.0

    for start in range(0, len(X), batch_size):
        indices = order[start:start + batch_size].to(X.device)

        X_batch = X[indices]
        y_batch = y[indices]

        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = loss_fn(predictions.reshape(-1), y_batch.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(indices)

    return total_loss / len(X)


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = X[start:start + batch_size]
            outputs = model(X_batch)

            predictions.append(outputs.reshape(-1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)

def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    predictions = predict(model, X_val)
    return rmse(y_val[ok], predictions[ok])


# ------------------------------------------------------------------ given

def train_mlp(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              device: str | None = None, tags: list[str] | None = None, log=print) -> dict:
    """Train on the pixel rows train_idx, validate on val_idx after every epoch, log everything.

    `X` is the pixel table, one row per row of `df`. Returns the history, the
    validation predictions after the first and the last epoch, the trained
    model, the checkpoint path and the W&B run page.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]

    # Belső tanító- és validációs halmaz
    inner_train, inner_val = build_splits(
        df.iloc[train_idx],
        group_column=cfg["split"]["group_column"],
        val_fraction=0.2,
        seed=cfg["split"]["seed"] + 1
    )

    # A relatív indexeket visszaalakítjuk az eredeti df indexeire
    inner_train_idx = train_idx[inner_train]
    inner_val_idx = train_idx[inner_val]
    
    # Standardizálás kizárólag a belső tanítóhalmaz alapján
    scaler = Standardiser().fit(X[inner_train_idx])
    
    # Belső tanítóadatok
    X_tr = to_tensor(scaler.transform(X[inner_train_idx]), device)
    y_tr = to_tensor(
        df[config.TARGET].to_numpy()[inner_train_idx], device
    )
    
    # Belső validációs adatok
    X_va = to_tensor(scaler.transform(X[inner_val_idx]), device)
    y_va = df[config.TARGET].to_numpy()[inner_val_idx]
    ok = df["eval_ok"].to_numpy()[inner_val_idx] == 1
    
    # Külső validációs adatok – csak a végső értékeléshez
    X_outer = to_tensor(scaler.transform(X[val_idx]), device)
    y_outer = df[config.TARGET].to_numpy()[val_idx]
    ok_outer = df["eval_ok"].to_numpy()[val_idx] == 1

    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=cfg["model"]["dropout"],
                prior=float(y_tr.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()           # the metric is RMSE; same minimiser
    generator = torch.Generator().manual_seed(cfg["seed"])

    tracking.init_run(cfg, tags=tags)
    history, snapshots = [], {}
    
    import copy
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 10
    epochs_without_improvement = 0
    log(f"{cfg['name']}: {len(inner_train_idx)} inner train / "
        f"{len(inner_val_idx)} inner val / {len(val_idx)} outer val rows, {X.shape[1]} inputs, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, {device}")
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, loss_fn, X_tr, y_tr, tr["batch_size"], generator)
        val = validate(model, X_va, y_va, ok)
        train_eval = rmse(y_tr.cpu().numpy(), predict(model, X_tr))   # dropout off, as in validation
        row = {"epoch": epoch, "train_loss": loss, "train_rmse": train_eval, "val_rmse": val,
               "seconds": time.time() - t0}
        history.append(row)
        tracking.log_metrics({"train_loss": loss, "train_rmse": train_eval, "val_rmse": val}, step=epoch)
        if epoch in (1, tr["epochs"]) or epoch % tr.get("log_every", 10) == 0:
            log(f"epoch {epoch:3d}  train loss {loss:.4f}  train RMSE {train_eval:.4f}  val RMSE {val:.4f}")
        if epoch in (1, tr["epochs"]):
            snapshots[epoch] = predict(model, X_va)
        # A legjobb modell kiválasztása a belső validáció alapján
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            log(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # A legjobb modell súlyainak visszaállítása
    if best_state is None:
        raise RuntimeError("No valid internal validation RMSE was obtained.")

    model.load_state_dict(best_state)

    # Végső értékelés a külső validációs halmazon
    outer_val_rmse = validate(model, X_outer, y_outer, ok_outer)

    log(f"Best epoch: {best_epoch}")
    log(f"Best internal val RMSE: {best_val:.4f}")
    log(f"Outer validation RMSE: {outer_val_rmse:.4f}")

    # A legjobb modell mentése
    ckpt = config.ROOT / "models" / f"{cfg['name']}.pt"
    ckpt.parent.mkdir(exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "in_dim": X.shape[1],
        "scaler_mean": scaler.mean,
        "scaler_scale": scaler.scale
    }, ckpt)

    tracking.log_checkpoint(
        ckpt,
        cfg["name"],
        {"val_rmse": outer_val_rmse, "best_epoch": best_epoch}
    )

    tracking.log_metrics({
        "final_val_rmse": outer_val_rmse,
        "best_internal_val_rmse": best_val,
        "best_epoch": best_epoch
    })

    url = tracking.run_url()
    tracking.finish()

    return {
        "history": pd.DataFrame(history),
        "val_pred": predict(model, X_outer),
        "snapshots": snapshots,
        "model": model,
        "checkpoint": ckpt,
        "url": url,
        "val_index": val_idx,
        "ok": ok_outer,
        "best_epoch": best_epoch,
        "best_internal_val_rmse": best_val,
        "outer_val_rmse": outer_val_rmse
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLP from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"], cfg["split"]["seed"])
    result = train_mlp(cfg, X, df, train_idx, val_idx)
    print(f"final val RMSE {result['history']['val_rmse'].iloc[-1]:.4f}  checkpoint {result['checkpoint']}  "
          f"run {result['url']}")


if __name__ == "__main__":
    main()
