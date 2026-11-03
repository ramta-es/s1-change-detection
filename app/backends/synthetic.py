"""Synthetic Sentinel-1-like scenes for tests and the no-credentials demo.

Generates a slowly varying land background with multiplicative gamma speckle, a permanent lake present
in both dates (must NOT be flagged), a flood that appears only in the post image, and a bright patch
that appears only in the post image (for the "increase" change type).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..processing.geometry import Grid
from .base import BackscatterStack, StackRequest


@dataclass(frozen=True)
class Truth:
    flood: np.ndarray  # boolean, newly flooded
    lake: np.ndarray  # boolean, permanent water
    bright: np.ndarray  # boolean, backscatter increase


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def make_truth(h: int, w: int) -> Truth:
    flood = _ellipse(h, w, 0.42 * h, 0.40 * w, 0.14 * h, 0.22 * w)
    lake = _ellipse(h, w, 0.78 * h, 0.80 * w, 0.08 * h, 0.09 * w)
    bright = _ellipse(h, w, 0.20 * h, 0.78 * w, 0.06 * h, 0.07 * w)
    flood &= ~lake
    return Truth(flood=flood, lake=lake, bright=bright)


def make_pair(grid: Grid, seed: int = 7, enl: float = 4.4) -> tuple[np.ndarray, np.ndarray, Truth]:
    h, w = grid.shape
    rng = np.random.default_rng(seed)
    truth = make_truth(h, w)

    # background sigma0 in dB: gentle spatial gradient + low-frequency texture
    yy, xx = np.mgrid[0:h, 0:w]
    base_db = -11.0 + 2.0 * np.sin(2 * np.pi * xx / max(w, 1)) * np.cos(2 * np.pi * yy / max(h, 1))
    base_db = base_db + rng.normal(0, 0.3, size=(h, w))  # per-pixel texture, identical on both dates

    pre_db = base_db.copy()
    post_db = base_db.copy()
    pre_db[truth.lake] = -22.0
    post_db[truth.lake] = -22.0
    post_db[truth.flood] = -19.0  # open water after the event
    post_db[truth.bright] = -4.0  # e.g. debris / double bounce

    def speckled(db: np.ndarray) -> np.ndarray:
        power = 10 ** (db / 10)
        # multiplicative speckle: gamma with shape = ENL, mean 1
        return (power * rng.gamma(enl, 1.0 / enl, size=db.shape)).astype(np.float32)

    pre = speckled(pre_db)
    post = speckled(post_db)
    # border nodata, as in real scene footprints
    pre[:2, :] = np.nan
    post[-2:, :] = np.nan
    return pre, post, truth


class SyntheticBackend:
    name = "synthetic"

    def __init__(self, seed: int = 7):
        self.seed = seed

    def fetch(self, req: StackRequest, workdir) -> BackscatterStack:
        seed = int(req.options.get("seed", self.seed))
        pre, post, truth = make_pair(req.grid, seed=seed)
        return BackscatterStack(
            pre=pre,
            post=post,
            grid=req.grid,
            n_pre_scenes=3,
            n_post_scenes=2,
            exclude_mask=None,
            metadata={"truth_flood_px": int(truth.flood.sum()), "truth_lake_px": int(truth.lake.sum())},
            warnings=["synthetic data: not a real observation"],
        )
