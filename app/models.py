"""Request / response models for the change-detection API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ChangeType = Literal["flood", "increase", "any"]
"""
flood     - backscatter DEcrease (open water is smooth -> dark in SAR)
increase  - backscatter INcrease (new structures, debris, vegetation flooding with double bounce)
any       - absolute change above threshold (generic ground disturbance)
"""


class DateWindow(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateWindow:
        if self.end < self.start:
            raise ValueError("window end must be on or after start")
        return self


class SpeckleParams(BaseModel):
    method: Literal["lee", "median", "none"] = "lee"
    window: int = Field(5, ge=3, le=15)

    @field_validator("window")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("speckle window must be odd")
        return v


class ChangeRequest(BaseModel):
    """One AOI (Polygon/MultiPolygon or a FeatureCollection of them), a pre window and a post window."""

    aoi: dict[str, Any] = Field(..., description="GeoJSON Polygon, MultiPolygon, Feature or FeatureCollection (WGS84)")
    pre: DateWindow
    post: DateWindow
    change_type: ChangeType = "flood"
    polarization: Literal["VV", "VH"] = "VV"
    orbit_state: Literal["ASCENDING", "DESCENDING"] | None = Field(
        None, description="Restrict to one orbit direction so pre and post share viewing geometry"
    )
    relative_orbit: int | None = Field(None, description="Restrict to one relative orbit (strongest geometry control)")
    threshold_db: float | None = Field(
        None,
        description="Change threshold in dB. None = Otsu on the difference image, with a sanity floor.",
    )
    speckle: SpeckleParams = SpeckleParams()
    min_mapping_unit_px: int = Field(8, ge=0, description="Drop connected components smaller than this many pixels")
    resolution_m: int | None = Field(None, ge=10, le=100)
    backend: Literal["cdse_openeo", "synthetic", "files"] | None = Field(
        None, description="Data backend override. Default comes from settings."
    )
    backend_options: dict[str, Any] = Field(
        default_factory=dict,
        description='Backend-specific options, e.g. {"pre_path": "...tif", "post_path": "...tif"} for the files backend',
    )

    @model_validator(mode="after")
    def _windows(self) -> ChangeRequest:
        if self.post.start <= self.pre.end:
            raise ValueError("post window must start after the pre window ends")
        return self


class PolygonStat(BaseModel):
    feature_id: str
    area_km2: float
    changed_area_km2: float
    changed_fraction: float


class ChangeResult(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    error: str | None = None
    backend: str | None = None
    crs: str | None = None
    resolution_m: float | None = None
    n_pre_scenes: int | None = None
    n_post_scenes: int | None = None
    threshold_db: float | None = None
    threshold_source: Literal["user", "otsu", "fallback"] | None = None
    aoi_area_km2: float | None = None
    valid_area_km2: float | None = None
    changed_area_km2: float | None = None
    changed_fraction: float | None = None
    polygons: list[PolygonStat] = []
    outputs: dict[str, str] = {}
    warnings: list[str] = []


class RunSummary(BaseModel):
    run_id: str
    status: str
    created_at: str
