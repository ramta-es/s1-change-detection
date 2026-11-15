"""Copernicus Data Space Ecosystem (CDSE) openEO backend.

One batch job per run produces a 4-band GeoTIFF on the processing grid:
    1 pre   - temporal median of backscatter (linear power) over the pre window
    2 post  - temporal median over the post window
    3 pre_count  - number of valid observations per pixel in the pre window
    4 post_count - same for the post window
Optionally a second, small synchronous request fetches ESA WorldCover and masks permanent water (class 80).

Why median composites: Sentinel-1 revisits every 6-12 days; a median over 2-4 scenes suppresses speckle
and single-date wind/rain artefacts far better than any spatial filter. The counts are reported so the
caller can see when a window only had one scene.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling

from ..config import settings
from ..processing.raster_io import read_to_grid
from .base import BackscatterStack, StackRequest

log = logging.getLogger(__name__)

COLLECTION = "SENTINEL1_GRD"
WORLDCOVER_COLLECTION = "ESA_WORLDCOVER_10M_2021"
WORLDCOVER_WATER_CLASS = 80

# Metadata property names used for load_collection filtering. Verify against
# connection.describe_collection("SENTINEL1_GRD")["summaries"] if a filter is silently ignored.
PROP_ORBIT_STATE = "sat:orbit_state"
PROP_RELATIVE_ORBIT = "relativeOrbitNumber"
PROP_INSTRUMENT_MODE = "sar:instrument_mode"


def _iso(d: date) -> str:
    return d.isoformat()


def _exclusive_end(d: date) -> str:
    # openEO temporal extents are [start, end) -> add one day to include `d`
    return (d + timedelta(days=1)).isoformat()


def default_connection_factory():
    import openeo

    conn = openeo.connect(settings.cdse_openeo_url)
    if settings.cdse_client_id and settings.cdse_client_secret:
        conn.authenticate_oidc_client_credentials(
            client_id=settings.cdse_client_id,
            client_secret=settings.cdse_client_secret,
            provider_id="CDSE",
        )
    else:
        # Device-code flow the first time (prints a URL), refresh token cached afterwards.
        conn.authenticate_oidc()
    return conn


class CdseOpenEOBackend:
    name = "cdse_openeo"

    def __init__(self, connection_factory: Callable[[], Any] | None = None):
        self._connect = connection_factory or default_connection_factory

    # ------------------------------------------------------------------ graph
    def build_cube(self, conn, req: StackRequest):
        """Build the 4-band process graph. Separated from fetch() so tests can inspect it."""
        west, south, east, north = req.bbox_wgs84
        props: dict[str, Any] = {PROP_INSTRUMENT_MODE: lambda m: m == "IW"}
        if req.orbit_state:
            props[PROP_ORBIT_STATE] = lambda s, _v=req.orbit_state: s == _v
        if req.relative_orbit is not None:
            props[PROP_RELATIVE_ORBIT] = lambda r, _v=req.relative_orbit: r == _v

        cube = conn.load_collection(
            COLLECTION,
            spatial_extent={"west": west, "south": south, "east": east, "north": north, "crs": "EPSG:4326"},
            temporal_extent=[_iso(req.pre[0]), _exclusive_end(req.post[1])],
            bands=[req.polarization],
            properties=props,
        )
        cube = cube.sar_backscatter(
            coefficient=req.options.get("coefficient", settings.cdse_backscatter_coefficient),
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=False,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
        )
        cube = cube.resample_spatial(
            resolution=req.grid.resolution_m,
            projection=req.grid.crs.to_epsg(),
            method="bilinear",
        )
        pol = req.polarization
        pre = cube.filter_temporal([_iso(req.pre[0]), _exclusive_end(req.pre[1])])
        post = cube.filter_temporal([_iso(req.post[0]), _exclusive_end(req.post[1])])

        pre_med = pre.reduce_dimension(dimension="t", reducer="median").rename_labels("bands", ["pre"], [pol])
        post_med = post.reduce_dimension(dimension="t", reducer="median").rename_labels("bands", ["post"], [pol])
        pre_cnt = pre.reduce_dimension(dimension="t", reducer="count").rename_labels("bands", ["pre_count"], [pol])
        post_cnt = post.reduce_dimension(dimension="t", reducer="count").rename_labels("bands", ["post_count"], [pol])
        return pre_med.merge_cubes(post_med).merge_cubes(pre_cnt).merge_cubes(post_cnt)

    def build_worldcover(self, conn, req: StackRequest):
        west, south, east, north = req.bbox_wgs84
        wc = conn.load_collection(
            WORLDCOVER_COLLECTION,
            spatial_extent={"west": west, "south": south, "east": east, "north": north, "crs": "EPSG:4326"},
            temporal_extent=["2021-01-01", "2021-12-31"],
            bands=["MAP"],
        )
        return wc.resample_spatial(resolution=req.grid.resolution_m, projection=req.grid.crs.to_epsg(), method="mode")

    # ------------------------------------------------------------------ fetch
    def fetch(self, req: StackRequest, workdir) -> BackscatterStack:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        warnings: list[str] = []

        cube = self.build_cube(conn, req)
        job = cube.create_job(title=f"s1-change-detection {req.pre[0]}..{req.post[1]}", out_format="GTiff")
        log.info("starting openEO batch job %s", getattr(job, "job_id", "?"))
        job.start_and_wait()
        files = [Path(p) for p in job.get_results().download_files(workdir / "openeo")]
        tifs = [p for p in files if p.suffix.lower() in (".tif", ".tiff")]
        if not tifs:
            raise RuntimeError(f"openEO job returned no GeoTIFF: {files}")
        stack_path = tifs[0]

        bands = self._band_index(stack_path)
        pre = read_to_grid_band(stack_path, bands["pre"], req.grid, Resampling.bilinear)
        post = read_to_grid_band(stack_path, bands["post"], req.grid, Resampling.bilinear)
        pre_cnt = read_to_grid_band(stack_path, bands["pre_count"], req.grid, Resampling.nearest)
        post_cnt = read_to_grid_band(stack_path, bands["post_count"], req.grid, Resampling.nearest)
        n_pre = int(np.nanmax(pre_cnt)) if np.isfinite(pre_cnt).any() else 0
        n_post = int(np.nanmax(post_cnt)) if np.isfinite(post_cnt).any() else 0
        if n_pre == 0 or n_post == 0:
            raise RuntimeError(f"no Sentinel-1 scenes in a window (pre={n_pre}, post={n_post}); widen the dates or drop the orbit filter")
        if n_pre == 1:
            warnings.append("pre window has a single scene: the composite carries that scene's speckle and weather")
        if n_post == 1:
            warnings.append("post window has a single scene")

        exclude = None
        if req.options.get("mask_permanent_water", settings.cdse_mask_permanent_water):
            try:
                wc_path = workdir / "worldcover.tif"
                self.build_worldcover(conn, req).download(wc_path, format="GTiff")
                wc = read_to_grid(wc_path, req.grid, resampling=Resampling.nearest)
                classes = np.where(np.isfinite(wc), np.rint(wc), -1).astype(int)
                exclude = classes == WORLDCOVER_WATER_CLASS
            except Exception as e:  # mask is a nice-to-have; do not fail the run
                warnings.append(f"WorldCover water mask unavailable ({e.__class__.__name__}); relying on pre-image darkness only")

        return BackscatterStack(
            pre=pre,
            post=post,
            grid=req.grid,
            n_pre_scenes=n_pre,
            n_post_scenes=n_post,
            exclude_mask=exclude,
            metadata={"job_id": getattr(job, "job_id", None), "stack_path": str(stack_path)},
            warnings=warnings,
        )

    @staticmethod
    def _band_index(path: Path) -> dict[str, int]:
        """Map band names to 1-based indices using GeoTIFF band descriptions, falling back to merge order."""
        order = ["pre", "post", "pre_count", "post_count"]
        with rasterio.open(path) as src:
            desc = [d or "" for d in src.descriptions]
            if src.count < 4:
                raise RuntimeError(f"expected 4 bands in {path.name}, found {src.count}")
        idx = {}
        for name in order:
            if name in desc:
                idx[name] = desc.index(name) + 1
        if len(idx) == 4:
            return idx
        return {name: i + 1 for i, name in enumerate(order)}


def read_to_grid_band(path: Path, band: int, grid, resampling: Resampling) -> np.ndarray:
    from rasterio.crs import CRS as RioCRS
    from rasterio.warp import reproject

    with rasterio.open(path) as src:
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
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
