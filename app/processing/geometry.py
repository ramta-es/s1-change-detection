"""AOI handling: GeoJSON in, shapely out, plus the projected CRS and grid used for processing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class AoiFeature:
    feature_id: str
    geom: Polygon | MultiPolygon  # WGS84


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in a projected CRS."""

    crs: CRS
    transform: Any  # affine.Affine
    width: int
    height: int
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return left, bottom, right, top


def parse_aoi(aoi: dict[str, Any]) -> list[AoiFeature]:
    """Accept a Polygon, MultiPolygon, Feature or FeatureCollection. Returns one feature per polygon input."""
    t = aoi.get("type")
    if t == "FeatureCollection":
        feats = aoi.get("features", [])
        if not feats:
            raise ValueError("FeatureCollection has no features")
        out = []
        for i, f in enumerate(feats):
            fid = str(f.get("id", f.get("properties", {}).get("id", i)))
            out.append(AoiFeature(fid, _as_polygonal(shape(f["geometry"]))))
        return out
    if t == "Feature":
        fid = str(aoi.get("id", aoi.get("properties", {}).get("id", 0)))
        return [AoiFeature(fid, _as_polygonal(shape(aoi["geometry"])))]
    if t in ("Polygon", "MultiPolygon"):
        return [AoiFeature("0", _as_polygonal(shape(aoi)))]
    raise ValueError(f"unsupported GeoJSON type: {t!r}")


def _as_polygonal(g) -> Polygon | MultiPolygon:
    if g.is_empty:
        raise ValueError("empty geometry")
    if not isinstance(g, (Polygon, MultiPolygon)):
        raise ValueError(f"geometry must be Polygon or MultiPolygon, got {g.geom_type}")
    if not g.is_valid:
        g = g.buffer(0)
    minx, miny, maxx, maxy = g.bounds
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
        raise ValueError("AOI coordinates must be WGS84 longitude/latitude")
    return g


def union_geom(features: list[AoiFeature]) -> Polygon | MultiPolygon:
    return _as_polygonal(unary_union([f.geom for f in features]))


def geodesic_area_km2(geom: Polygon | MultiPolygon) -> float:
    """Ellipsoidal area of a WGS84 polygon, in km²."""
    area_m2, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area_m2) / 1e6


def utm_crs_for(geom: Polygon | MultiPolygon) -> CRS:
    """UTM zone (WGS84) containing the AOI centroid. Good enough for AOIs up to a few hundred km across."""
    lon, lat = geom.centroid.x, geom.centroid.y
    zone = int(math.floor((lon + 180) / 6) + 1)
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def project_geom(geom: Polygon | MultiPolygon, crs: CRS) -> Polygon | MultiPolygon:
    """WGS84 geometry -> `crs` (vectorised, shapely 2)."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _f(coords: np.ndarray) -> np.ndarray:
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, _f)


def make_grid(geom: Polygon | MultiPolygon, crs: CRS, resolution_m: float, pad_px: int = 2) -> Grid:
    """Grid covering the AOI's projected bounds, snapped to the resolution."""
    minx, miny, maxx, maxy = project_geom(geom, crs).bounds
    res = float(resolution_m)
    left = math.floor(minx / res) * res - pad_px * res
    top = math.ceil(maxy / res) * res + pad_px * res
    right = math.ceil(maxx / res) * res + pad_px * res
    bottom = math.floor(miny / res) * res - pad_px * res
    width = round((right - left) / res)
    height = round((top - bottom) / res)
    return Grid(crs=crs, transform=from_origin(left, top, res, res), width=width, height=height, resolution_m=res)


def pixel_area_km2(grid: Grid) -> np.ndarray:
    """Per-pixel area as an (H, W) array in km². Exact for projected grids; latitude-corrected for EPSG:4326."""
    if grid.crs.is_projected:
        return np.full(grid.shape, (grid.resolution_m**2) / 1e6, dtype=np.float64)
    # geographic grid: pixel size is in degrees
    a, _, _, _, e, f = grid.transform.a, grid.transform.b, grid.transform.c, grid.transform.d, grid.transform.e, grid.transform.f
    dlon, dlat = abs(a), abs(e)
    rows = np.arange(grid.height) + 0.5
    lat = f + rows * e
    m_per_deg = 111_320.0
    row_area = (dlat * m_per_deg) * (dlon * m_per_deg * np.cos(np.radians(lat))) / 1e6
    return np.repeat(row_area[:, None], grid.width, axis=1)
