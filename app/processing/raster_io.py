"""GeoTIFF / PNG output and GeoTIFF input helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS as RioCRS
from rasterio.warp import Resampling, reproject

from .geometry import Grid

_COG_PROFILE = {"driver": "GTiff", "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate", "predictor": 2}


def write_geotiff(path: Path, arr: np.ndarray, grid: Grid, nodata=None, dtype=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = dtype or arr.dtype
    profile = dict(
        _COG_PROFILE,
        height=grid.height,
        width=grid.width,
        count=1,
        dtype=np.dtype(dtype).name,
        crs=RioCRS.from_wkt(grid.crs.to_wkt()),
        transform=grid.transform,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
        # internal overviews make the file behave like a COG for viewers
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest if np.issubdtype(dtype, np.integer) else Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="nearest")
    return path


def write_preview_png(path: Path, post_db: np.ndarray, mask: np.ndarray, inside: np.ndarray | None = None) -> Path:
    """Grayscale post-event backscatter with changed pixels in red and outside-AOI dimmed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(post_db)
    lo, hi = (np.nanpercentile(post_db, [2, 98]) if valid.any() else (-25.0, 0.0))
    g = np.clip((post_db - lo) / max(hi - lo, 1e-6), 0, 1)
    g = np.where(valid, g, 0.0)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[mask == 1] = [1.0, 0.15, 0.1]
    if inside is not None:
        rgb[~inside] *= 0.45
    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    # keep previews small
    max_side = 1600
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)
    img.save(path)
    return path


def read_to_grid(path: Path, grid: Grid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Read band 1 of a GeoTIFF and reproject it onto `grid`. Returns float32 with NaN nodata."""
    with rasterio.open(path) as src:
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=RioCRS.from_wkt(grid.crs.to_wkt()),
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out
