"""The CDSE backend is exercised against a fake openEO connection: no network, no credentials."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS as RioCRS

from app.backends.base import StackRequest
from app.backends.openeo_cdse import (
    COLLECTION,
    PROP_INSTRUMENT_MODE,
    PROP_ORBIT_STATE,
    PROP_RELATIVE_ORBIT,
    WORLDCOVER_COLLECTION,
    CdseOpenEOBackend,
)
from app.backends.synthetic import make_pair
from app.processing.change import detect_change


class FakeCube:
    def __init__(self, log, collection):
        self.log = log
        self.collection = collection

    def _rec(self, name, *a, **kw):
        self.log.append((name, a, kw))
        c = FakeCube(self.log, self.collection)
        c.log_grid = getattr(self, "log_grid", None)
        return c

    def sar_backscatter(self, **kw):
        return self._rec("sar_backscatter", **kw)

    def resample_spatial(self, **kw):
        return self._rec("resample_spatial", **kw)

    def filter_temporal(self, extent):
        return self._rec("filter_temporal", extent)

    def reduce_dimension(self, dimension, reducer):
        return self._rec("reduce_dimension", dimension=dimension, reducer=reducer)

    def rename_labels(self, dimension, target, source=None):
        return self._rec("rename_labels", dimension=dimension, target=target, source=source)

    def merge_cubes(self, other):
        return self._rec("merge_cubes")

    def create_job(self, title, out_format):
        self.log.append(("create_job", (title, out_format), {}))
        return FakeJob(self.log)

    def download(self, path, format):
        # WorldCover: write a tiny MAP raster with class 80 in the top-left corner
        grid = self.log_grid
        arr = np.full(grid.shape, 10, dtype=np.uint8)
        arr[: grid.height // 10, : grid.width // 10] = 80
        _write(Path(path), [arr], grid, ["MAP"], np.uint8)


class FakeJob:
    job_id = "fake-job-1"

    def __init__(self, log):
        self.log = log
        self.started = False

    def start_and_wait(self):
        self.started = True
        self.log.append(("start_and_wait", (), {}))

    def get_results(self):
        return self

    def download_files(self, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        grid = FakeConnection.last_grid
        pre, post, _ = make_pair(grid, seed=11)
        pre_cnt = np.full(grid.shape, 3, dtype=np.float32)
        post_cnt = np.full(grid.shape, 2, dtype=np.float32)
        p = target / "openEO.tif"
        _write(p, [pre, post, pre_cnt, post_cnt], grid, ["pre", "post", "pre_count", "post_count"], np.float32)
        return [p]


class FakeConnection:
    last_grid = None

    def __init__(self):
        self.log = []

    def load_collection(self, collection, **kw):
        self.log.append(("load_collection", (collection,), kw))
        cube = FakeCube(self.log, collection)
        cube.log_grid = FakeConnection.last_grid
        return cube


def _write(path: Path, bands, grid, names, dtype):
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.height, width=grid.width, count=len(bands), dtype=np.dtype(dtype).name,
        crs=RioCRS.from_wkt(grid.crs.to_wkt()), transform=grid.transform, nodata=None,
    ) as dst:
        for i, (b, n) in enumerate(zip(bands, names, strict=True), start=1):
            dst.write(np.nan_to_num(b, nan=0.0).astype(dtype), i)
            dst.set_band_description(i, n)


def _request(grid, **kw) -> StackRequest:
    return StackRequest(
        grid=grid, bbox_wgs84=(-0.48, 39.30, -0.30, 39.46),
        pre=(date(2024, 10, 5), date(2024, 10, 28)), post=(date(2024, 10, 30), date(2024, 11, 12)),
        polarization="VV", **kw,
    )


def test_process_graph_filters_and_bands(small_grid):
    FakeConnection.last_grid = small_grid
    conn = FakeConnection()
    be = CdseOpenEOBackend(connection_factory=lambda: conn)
    be.build_cube(conn, _request(small_grid, orbit_state="DESCENDING", relative_orbit=110))

    names = [e[0] for e in conn.log]
    assert names[0] == "load_collection"
    lc = conn.log[0]
    assert lc[1] == (COLLECTION,)
    assert lc[2]["bands"] == ["VV"]
    assert lc[2]["temporal_extent"] == ["2024-10-05", "2024-11-13"]  # end is exclusive -> +1 day
    props = lc[2]["properties"]
    assert props[PROP_INSTRUMENT_MODE]("IW") and not props[PROP_INSTRUMENT_MODE]("EW")
    assert props[PROP_ORBIT_STATE]("DESCENDING") and not props[PROP_ORBIT_STATE]("ASCENDING")
    assert props[PROP_RELATIVE_ORBIT](110) and not props[PROP_RELATIVE_ORBIT](8)

    assert names.count("sar_backscatter") == 1
    rs = next(e for e in conn.log if e[0] == "resample_spatial")
    assert rs[2]["resolution"] == small_grid.resolution_m and rs[2]["projection"] == 32630
    assert names.count("filter_temporal") == 2
    reducers = [e[2]["reducer"] for e in conn.log if e[0] == "reduce_dimension"]
    assert sorted(reducers) == ["count", "count", "median", "median"]
    targets = [e[2]["target"][0] for e in conn.log if e[0] == "rename_labels"]
    assert targets == ["pre", "post", "pre_count", "post_count"]
    assert names.count("merge_cubes") == 3


def test_no_orbit_filters_when_not_requested(small_grid):
    conn = FakeConnection()
    CdseOpenEOBackend(connection_factory=lambda: conn).build_cube(conn, _request(small_grid))
    props = conn.log[0][2]["properties"]
    assert PROP_ORBIT_STATE not in props and PROP_RELATIVE_ORBIT not in props


def test_fetch_reads_stack_counts_and_water_mask(small_grid, tmp_path):
    FakeConnection.last_grid = small_grid
    conn = FakeConnection()
    be = CdseOpenEOBackend(connection_factory=lambda: conn)
    stack = be.fetch(_request(small_grid), workdir=tmp_path)

    assert stack.n_pre_scenes == 3 and stack.n_post_scenes == 2
    assert stack.pre.shape == small_grid.shape and np.isfinite(stack.pre).mean() > 0.9
    assert stack.exclude_mask is not None and stack.exclude_mask[0, 0] and not stack.exclude_mask[-1, -1]
    assert stack.metadata["job_id"] == "fake-job-1"
    assert any(e[0] == "load_collection" and e[1] == (WORLDCOVER_COLLECTION,) for e in conn.log)

    # and the stack is usable downstream: the synthetic flood is found
    out = detect_change(stack.pre, stack.post, change_type="flood", exclude_mask=stack.exclude_mask)
    assert out.threshold_source == "otsu" and (out.mask == 1).sum() > 0
    assert np.all(out.mask[stack.exclude_mask] == 255)


def test_fetch_tolerates_missing_worldcover(small_grid, tmp_path, monkeypatch):
    FakeConnection.last_grid = small_grid
    conn = FakeConnection()
    be = CdseOpenEOBackend(connection_factory=lambda: conn)

    def boom(*a, **kw):
        raise RuntimeError("worldcover down")

    monkeypatch.setattr(be, "build_worldcover", boom)
    stack = be.fetch(_request(small_grid), workdir=tmp_path)
    assert stack.exclude_mask is None
    assert any("WorldCover" in w for w in stack.warnings)


def test_fetch_fails_when_a_window_is_empty(small_grid, tmp_path, monkeypatch):
    FakeConnection.last_grid = small_grid
    conn = FakeConnection()

    def empty_download(self, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        z = np.zeros(small_grid.shape, dtype=np.float32)
        p = target / "openEO.tif"
        _write(p, [z, z, z, z], small_grid, ["pre", "post", "pre_count", "post_count"], np.float32)
        return [p]

    monkeypatch.setattr(FakeJob, "download_files", empty_download)
    with pytest.raises(RuntimeError, match="no Sentinel-1 scenes"):
        CdseOpenEOBackend(connection_factory=lambda: conn).fetch(_request(small_grid), workdir=tmp_path)
