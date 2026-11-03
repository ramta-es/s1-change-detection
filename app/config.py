"""Runtime settings. Everything is overridable through environment variables (see .env.example)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="S1CD_", env_file=".env", extra="ignore")

    # Where run outputs (GeoTIFF, PNG, JSON) are written. One sub-directory per run id.
    runs_dir: Path = Path("runs")

    # Default data backend. "cdse_openeo" needs Copernicus Data Space credentials.
    # "synthetic" needs nothing and is used by the tests and the 2-minute demo.
    default_backend: Literal["cdse_openeo", "synthetic", "files"] = "cdse_openeo"

    # Optional PostGIS persistence. When unset, runs are kept in memory.
    database_url: str | None = None

    # Copernicus Data Space Ecosystem openEO
    cdse_openeo_url: str = "https://openeo.dataspace.copernicus.eu"
    # Client-credentials flow (register an OIDC client in the CDSE portal). If both are empty
    # the service falls back to the interactive device-code flow with a cached refresh token.
    cdse_client_id: str | None = None
    cdse_client_secret: str | None = None
    # "sigma0-ellipsoid" is always available on CDSE; "gamma0-terrain" is better in hilly AOIs.
    cdse_backscatter_coefficient: str = "sigma0-ellipsoid"
    # Mask ESA WorldCover permanent water (class 80) when running on CDSE.
    cdse_mask_permanent_water: bool = True

    # Processing defaults
    resolution_m: int = 20
    max_aoi_km2: float = 5000.0

    # Run POST /change synchronously (useful in tests and CLI). The API also honours ?wait=true.
    sync: bool = False


settings = Settings()
