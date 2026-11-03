from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.processing.geometry import make_grid, parse_aoi, union_geom, utm_crs_for

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def _runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    monkeypatch.setattr(settings, "default_backend", "synthetic")
    monkeypatch.setattr(settings, "sync", False)
    yield tmp_path / "runs"


@pytest.fixture
def aoi_fc() -> dict:
    return json.loads((EXAMPLES / "aoi_valencia.geojson").read_text())


@pytest.fixture
def request_json() -> dict:
    return json.loads((EXAMPLES / "request_synthetic.json").read_text())


@pytest.fixture
def small_grid(aoi_fc):
    aoi = union_geom(parse_aoi(aoi_fc))
    return make_grid(aoi, utm_crs_for(aoi), 80)  # ~230 x 200 px, fast
