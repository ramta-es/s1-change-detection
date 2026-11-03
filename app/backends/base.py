"""Backend protocol: anything that can produce a co-registered pre/post backscatter pair on a Grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import numpy as np

from ..processing.geometry import Grid


@dataclass
class StackRequest:
    grid: Grid
    bbox_wgs84: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    pre: tuple[date, date]
    post: tuple[date, date]
    polarization: str = "VV"
    orbit_state: str | None = None
    relative_orbit: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackscatterStack:
    """Linear-power (sigma0 or gamma0) composites on `grid`. NaN = nodata."""

    pre: np.ndarray
    post: np.ndarray
    grid: Grid
    n_pre_scenes: int | None = None
    n_post_scenes: int | None = None
    exclude_mask: np.ndarray | None = None  # True where never to flag change (permanent water, layover)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def fetch(self, req: StackRequest, workdir) -> BackscatterStack: ...
