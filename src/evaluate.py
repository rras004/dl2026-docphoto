"""Score predictions and break the error down.

RMSE is the competition metric, but a single number will not tell you what to
fix. Slice the error by document family, language, font, font size and
photographer to see where your model actually fails, and leave `eval_ok = 0`
rows out of every number you report.

Lab 4 adds three things to lab 3's file:

* `mae`, the error a person reads more easily: "off by 0.1 on average";
* `page_bootstrap`, which answers "is model B really better than model A on
  these validation photographs?" by resampling whole pages, because the
  photographs of one page are not independent samples;
* `regression_report`, one figure and one table that you should look at for
  every model you train: predicted against true, the residuals against the
  prediction, the calibration of the prediction, and the residuals by
  category, next to the same numbers for the training-mean baseline.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def by_slice(df: pd.DataFrame, pred: np.ndarray, column: str) -> pd.DataFrame:
    """RMSE per value of `column`, with the count and the mean target and prediction."""
    d = pd.DataFrame({column: df[column].to_numpy(), "y": df[config.TARGET].to_numpy(), "p": pred})
    rows = []
    for key, g in d.groupby(column):
        rows.append({column: key, "n": len(g), "rmse": rmse(g["y"], g["p"]),
                     "mean_target": g["y"].mean(), "mean_pred": g["p"].mean()})
    return pd.DataFrame(rows).set_index(column).sort_values("rmse", ascending=False)


def worst(df: pd.DataFrame, pred: np.ndarray, k: int = 5) -> pd.DataFrame:
    """The k rows with the largest absolute error, with the columns worth looking at."""
    d = df.copy()
    d["pred"] = pred
    d["error"] = d["pred"] - d[config.TARGET]
    cols = ["image_id", config.TARGET, "pred", "error", "family", "font", "size_pt", "photographer"]
    return d.reindex(d["error"].abs().sort_values(ascending=False).index)[cols].head(k)


# ---------------------------------------------------------------- comparing two models

def page_bootstrap(y, pred_a, pred_b, pages, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over pages: how sure are we that B beats A on these photographs?

    Draw as many pages as there are, with replacement, keep every photograph of
    each drawn page, and recompute both RMSEs on the same draw. Repeat `n`
    times. The spread of `rmse_b - rmse_a` over the draws is the uncertainty
    of the comparison; the share of draws in which B is lower is how often B
    wins. Pairing (the same draw for both models) is what makes small gaps
    resolvable: most of the noise is in which pages are hard, and that noise
    hits both models alike.

    Returns rmse_a, rmse_b, diff (b - a), lo and hi (the 95 % interval of the
    difference), and share_b_better.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    d = pd.DataFrame({"page": np.asarray(pages), "ea": (a - y) ** 2, "eb": (b - y) ** 2})
    by = d.groupby("page").agg(ea=("ea", "sum"), eb=("eb", "sum"), n=("ea", "size"))
    ea, eb, cnt = by["ea"].to_numpy(), by["eb"].to_numpy(), by["n"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(by), size=(n, len(by)))
    ra = np.sqrt(ea[k].sum(1) / cnt[k].sum(1))
    rb = np.sqrt(eb[k].sum(1) / cnt[k].sum(1))
    diff = rb - ra
    return {"rmse_a": rmse(y, a), "rmse_b": rmse(y, b), "diff": rmse(y, b) - rmse(y, a),
            "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "share_b_better": float(np.mean(diff < 0)), "pages": int(len(by))}


# ---------------------------------------------------------------- the regression report

def regression_table(y, pred, baseline: float | None = None) -> pd.DataFrame:
    """RMSE, MAE, bias, spread ratio and Spearman for the model and, if given, a constant baseline.

    * bias = mean(pred - y): above zero, the model is too optimistic overall;
    * spread ratio = sd(pred) / sd(y): below one, the model pulls its
      predictions towards the mean and does not dare to call a photograph bad.
      A constant predictor has spread ratio 0. A model that has learned
      nothing but is trained on squared error drifts towards 0 too;
    * Spearman: does the model order the photographs correctly, regardless of
      the scale of its numbers.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    rows = {"model": pred}
    if baseline is not None:
        rows["train mean"] = np.full_like(y, float(baseline))
    out = {}
    for name, p in rows.items():
        rho = spearmanr(y, p).statistic if np.ptp(p) > 0 else float("nan")
        out[name] = {"rmse": rmse(y, p), "mae": mae(y, p), "bias": float(np.mean(p - y)),
                     "spread ratio": float(np.std(p) / np.std(y)), "spearman": float(rho)}
    return pd.DataFrame(out).T


def regression_report(df: pd.DataFrame, y, pred, baseline: float | None = None, by: str = "family",
                      title: str | None = None, bins: int = 10):
    """The table above and a four-panel figure. Pass only the rows you score (eval_ok == 1).

    1. predicted against true, coloured by `by`, with the line y = x and the
       baseline. Points on a horizontal band = the model ignores the image;
       points on the diagonal = right.
    2. residual (pred - true) against the prediction, with the mean residual
       per bin of the prediction. A trend here is something the model could
       still learn.
    3. calibration: the mean true value per bin of the prediction against the
       mean prediction in that bin. On the diagonal = when the model says 0.6,
       the photographs are worth 0.6 on average. Marker size = photographs.
    4. residuals by `by`, one box per category: which kind of document the
       model is systematically too optimistic or too pessimistic about.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    table = regression_table(y, pred, baseline)
    cat = df[by].astype(str).to_numpy()
    cats = sorted(set(cat))
    colors = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cats)}
    res = pred - y

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    for c in cats:
        m = cat == c
        ax[0].scatter(y[m], pred[m], s=9, alpha=0.6, color=colors[c], label=f"{c} ({m.sum()})")
    ax[0].plot([0, 1], [0, 1], color="k", lw=0.8)
    if baseline is not None:
        ax[0].axhline(baseline, color="0.5", ls="--", lw=0.8, label="train mean")
    ax[0].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="true usability", ylabel="predicted",
              title="predicted vs true")
    ax[0].legend(fontsize=7, loc="upper left")

    edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, max(len(edges) - 2, 0))
    groups = [np.flatnonzero(which == b) for b in range(max(len(edges) - 1, 1))]
    groups = [g for g in groups if len(g)]
    centre = np.array([pred[g].mean() for g in groups])
    ax[1].scatter(pred, res, s=7, alpha=0.4, color="0.4")
    ax[1].plot(centre, [res[g].mean() for g in groups], "o-", color="C3", label="mean per bin")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="predicted", ylabel="residual (pred - true)", title="residual vs predicted")
    ax[1].legend(fontsize=7)

    ax[2].plot([0, 1], [0, 1], color="k", lw=0.8)
    ax[2].scatter(centre, [y[g].mean() for g in groups], s=[12 + 3 * len(g) ** 0.8 for g in groups], color="C0")
    ax[2].plot(centre, [y[g].mean() for g in groups], color="C0", lw=1)
    ax[2].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="mean prediction in bin",
              ylabel="mean true value in bin", title="calibration (regression)")

    ax[3].boxplot([res[cat == c] for c in cats], showfliers=False)
    ax[3].set_xticks(range(1, len(cats) + 1), cats, rotation=30, fontsize=8)
    ax[3].axhline(0, color="k", lw=0.8)
    ax[3].set(ylabel="residual (pred - true)", title=f"residuals by {by}")

    t = table.loc["model"]
    fig.suptitle((title + "   " if title else "") + f"RMSE {t['rmse']:.4f}  MAE {t['mae']:.4f}  "
                 f"bias {t['bias']:+.3f}  spread ratio {t['spread ratio']:.2f}  Spearman {t['spearman']:.3f}",
                 y=1.03, fontsize=10)
    fig.tight_layout()
    return table, fig
