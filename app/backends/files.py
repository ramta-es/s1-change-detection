"""Local-files backend: pre and post GeoTIFFs you already have (e.g. from SNAP, ASF or a previous openEO job)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.warp import Resampling

from ..processing.raster_io import read_to_grid
from .base import BackscatterStack, StackRequest


class FilesBackend:
    """options: pre_path, post_path (required), input_in_db (default False), water_mask_path (optional, >0 = water)."""

    name = "files"

    def fetch(self, req: StackRequest, workdir) -> BackscatterStack:
        opts = req.options
        try:
            pre_path, post_path = Path(opts["pre_path"]), Path(opts["post_path"])
        except KeyError as e:
            raise ValueError(f"files backend needs backend_options.{e.args[0]}") from None
        for p in (pre_path, post_path):
            if not p.exists():
                raise FileNotFoundError(p)

        pre = read_to_grid(pre_path, req.grid)
        post = read_to_grid(post_path, req.grid)
        if opts.get("input_in_db", False):
            pre = (10 ** (pre / 10)).astype(np.float32)
            post = (10 ** (post / 10)).astype(np.float32)

        exclude = None
        wm = opts.get("water_mask_path")
        if wm:
            water = read_to_grid(Path(wm), req.grid, resampling=Resampling.nearest)
            exclude = np.isfinite(water) & (water > 0)

        return BackscatterStack(
            pre=pre,
            post=post,
            grid=req.grid,
            exclude_mask=exclude,
            metadata={"pre_path": str(pre_path), "post_path": str(post_path)},
        )
