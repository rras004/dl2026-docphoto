"""What a model receives: a table of inputs for training, one tensor for a single image.

Given. Read it once; it is the reason your validation score and your
leaderboard score can be compared at all.

Training reads the 768 px cache (`data/work/train_768/`), because decoding the
2048 px originals every time is slow. A test photograph has no cache entry.
If prediction resized it some other way, the network would see slightly
different pixels than it was trained on, and that difference is invisible in
your validation score and visible on the leaderboard. `as_cached` therefore
rebuilds in memory exactly the bytes the cache holds for an image (same
decoder shortcut, same resampling, same JPEG quality), and `preprocess_one`
runs the same steps as the training table from there.

Lab 4 knows one kind of input, `grey`: the lab-3 pixels at a larger
size, as one-channel images stored as bytes. Lab 5 adds colour images for
pretrained networks and their embeddings.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import WORK, cache_dir, pixels_of_image

CACHE_SIZE = 768


def as_cached(path: str | Path, size: int = CACHE_SIZE) -> BytesIO:
    """The bytes `build_cache` would have written for this photograph, without writing them."""
    im = Image.open(path)
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf


def grey_table(df: pd.DataFrame, size: int, workers: int = 4) -> np.ndarray:
    """All photographs as uint8 grey images of shape (rows, 1, size, size), rows in the order of `df`.

    The same pixels as lab 3's `pixel_table` (long side scaled to `size`,
    mid-grey padding to a square), stored as bytes: at 512 px a float table
    would take 1.9 GB, the bytes take a quarter of that. Nothing is lost,
    since the resampled image is 8-bit to begin with; the training loop
    turns a batch into floats when it needs it. Cached in
    data/work/grey_<size>.npy.
    """
    store = WORK / f"grey_{size}.npy"
    ids_store = WORK / f"grey_{size}_ids.npy"
    ids = df["image_id"].to_numpy().astype(str)
    if store.exists() and ids_store.exists() and (np.load(ids_store) == ids).all():
        return np.load(store)
    source = cache_dir(CACHE_SIZE)
    paths = [source / f"{i}.jpg" if (source / f"{i}.jpg").exists() else p
             for i, p in zip(df["image_id"], df["path"], strict=True)]

    def one(p):
        return np.rint(pixels_of_image(p, size) * 255).astype(np.uint8).reshape(1, size, size)

    with ThreadPoolExecutor(workers) as ex:
        x = np.stack(list(ex.map(one, paths)))
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(store, x)
    np.save(ids_store, ids)
    return x


def load_inputs(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """The whole input table for `df`, rows in the order of `df`.

    grey: uint8 of shape (rows, 1, size, size); the training loop scales it
    to [0, 1] and standardises it with statistics of the training rows.
    """
    kind = cfg["data"].get("kind", "grey")
    if kind == "grey":
        return grey_table(df, cfg["data"]["pixel_size"])
    raise ValueError(f"unknown input kind {kind!r}")


def standardise_inputs(cfg: dict) -> bool:
    """Grey pixels are standardised with the training rows' mean and scale, as in lab 3."""
    return cfg["data"].get("kind", "grey") == "grey"


def preprocess_one(image_path: str | Path, cfg: dict, checkpoint: dict) -> torch.Tensor:
    """One photograph as a batch of one, exactly as the training table and the loop prepared it."""
    kind = cfg["data"].get("kind", "grey")
    if kind == "grey":
        s = cfg["data"]["pixel_size"]
        x = pixels_of_image(as_cached(image_path), s).reshape(1, 1, s, s)
        sc = checkpoint["scaler"]
        return torch.from_numpy(((x - sc["mean"]) / sc["scale"]).astype(np.float32))
    raise ValueError(f"unknown input kind {kind!r}")


def main() -> None:
    import argparse

    from src.data.dataset import build_cache, load_table

    ap = argparse.ArgumentParser(description="Build the lab-4 input tables under data/work/")
    ap.add_argument("--grey", type=int, action="append", help="grey table size; repeatable (default 256 and 768)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = load_table()
    build_cache(df, CACHE_SIZE, a.workers)
    for size in a.grey or [256, 768]:
        x = grey_table(df, size, a.workers)
        print(f"grey table {size} px: {x.shape}, {x.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
