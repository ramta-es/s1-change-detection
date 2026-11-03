"""Area statistics for the whole AOI and per input polygon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize

from .geometry import AoiFeature, Grid, pixel_area_km2, project_geom


@dataclass(frozen=True)
class AreaStats:
    aoi_area_km2: float
    valid_area_km2: float
    changed_area_km2: float
    changed_fraction: float  # of valid area


def rasterize_features(features: list[AoiFeature], grid: Grid) -> np.ndarray:
    """int32 label raster: pixel value = 1-based index into `features`, 0 outside all polygons."""
    shapes = [(project_geom(f.geom, grid.crs), i + 1) for i, f in enumerate(features)]
    return rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )


def area_stats(mask: np.ndarray, inside: np.ndarray, grid: Grid) -> AreaStats:
    """mask: uint8 (1 changed, 0 unchanged, 255 nodata). inside: boolean, pixels belonging to the AOI."""
    px = pixel_area_km2(grid)
    aoi_area = float(px[inside].sum())
    valid = inside & (mask != 255)
    valid_area = float(px[valid].sum())
    changed = inside & (mask == 1)
    changed_area = float(px[changed].sum())
    frac = changed_area / valid_area if valid_area > 0 else 0.0
    return AreaStats(aoi_area, valid_area, changed_area, frac)


def per_polygon_stats(mask: np.ndarray, labels: np.ndarray, features: list[AoiFeature], grid: Grid):
    out = []
    for i, f in enumerate(features):
        inside = labels == (i + 1)
        s = area_stats(mask, inside, grid)
        out.append((f.feature_id, s))
    return out
