"""End-to-end run: request -> grid -> backend -> change detection -> stats -> files -> result."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .backends import StackRequest, get_backend
from .config import settings
from .models import ChangeRequest, ChangeResult, PolygonStat
from .processing.change import detect_change
from .processing.geometry import (
    geodesic_area_km2,
    make_grid,
    parse_aoi,
    union_geom,
    utm_crs_for,
)
from .processing.raster_io import write_geotiff, write_preview_png
from .processing.stats import area_stats, per_polygon_stats, rasterize_features

log = logging.getLogger(__name__)


def run_change(req: ChangeRequest, run_id: str, runs_dir: Path | None = None) -> ChangeResult:
    t0 = time.time()
    runs_dir = Path(runs_dir or settings.runs_dir)
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    features = parse_aoi(req.aoi)
    aoi = union_geom(features)
    aoi_km2 = geodesic_area_km2(aoi)
    if aoi_km2 > settings.max_aoi_km2:
        raise ValueError(f"AOI is {aoi_km2:.0f} km²; the limit is {settings.max_aoi_km2:.0f} km²")

    res = req.resolution_m or settings.resolution_m
    crs = utm_crs_for(aoi)
    grid = make_grid(aoi, crs, res)
    if grid.width * grid.height > 60_000_000:
        raise ValueError("grid too large; raise resolution_m or shrink the AOI")

    backend_name = req.backend or settings.default_backend
    backend = get_backend(backend_name)
    stack = backend.fetch(
        StackRequest(
            grid=grid,
            bbox_wgs84=aoi.bounds,
            pre=(req.pre.start, req.pre.end),
            post=(req.post.start, req.post.end),
            polarization=req.polarization,
            orbit_state=req.orbit_state,
            relative_orbit=req.relative_orbit,
            options=req.backend_options,
        ),
        workdir=out_dir / "work",
    )

    change = detect_change(
        stack.pre,
        stack.post,
        change_type=req.change_type,
        threshold_db=req.threshold_db,
        speckle_method=req.speckle.method,
        speckle_window=req.speckle.window,
        min_mapping_unit_px=req.min_mapping_unit_px,
        exclude_mask=stack.exclude_mask,
    )

    labels = rasterize_features(features, grid)
    inside = labels > 0
    mask = change.mask.copy()
    mask[~inside] = 255  # report only inside the AOI
    totals = area_stats(mask, inside, grid)
    polys = [
        PolygonStat(feature_id=fid, area_km2=s.aoi_area_km2, changed_area_km2=s.changed_area_km2, changed_fraction=s.changed_fraction)
        for fid, s in per_polygon_stats(mask, labels, features, grid)
    ]

    outputs = {
        "mask_tif": str(write_geotiff(out_dir / "mask.tif", mask, grid, nodata=255, dtype=np.uint8)),
        "diff_tif": str(write_geotiff(out_dir / "diff_db.tif", np.where(inside, change.diff_db, np.nan), grid, nodata=np.nan, dtype=np.float32)),
        "preview_png": str(write_preview_png(out_dir / "preview.png", change.post_db, mask, inside)),
    }

    result = ChangeResult(
        run_id=run_id,
        status="completed",
        backend=backend_name,
        crs=f"EPSG:{crs.to_epsg()}",
        resolution_m=float(res),
        n_pre_scenes=stack.n_pre_scenes,
        n_post_scenes=stack.n_post_scenes,
        threshold_db=change.threshold_db,
        threshold_source=change.threshold_source,  # type: ignore[arg-type]
        aoi_area_km2=round(totals.aoi_area_km2, 4),
        valid_area_km2=round(totals.valid_area_km2, 4),
        changed_area_km2=round(totals.changed_area_km2, 4),
        changed_fraction=round(totals.changed_fraction, 5),
        polygons=polys,
        outputs=outputs,
        warnings=[*stack.warnings, *change.warnings],
    )
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "request.json").write_text(req.model_dump_json(indent=2))
    log.info("run %s done in %.1fs: %.3f km² changed (%.1f%%)", run_id, time.time() - t0, totals.changed_area_km2, 100 * totals.changed_fraction)
    return result
