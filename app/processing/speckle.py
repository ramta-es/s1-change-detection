"""Speckle filtering on linear-power backscatter. Applied before the dB conversion."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def lee_filter(img: np.ndarray, window: int = 5, enl: float = 4.4) -> np.ndarray:
    """Classic Lee (1980) adaptive filter.

    Smooths homogeneous areas toward the local mean and keeps edges where local variance is high.
    `enl` is the equivalent number of looks of the input; ~4.4 for Sentinel-1 GRD IW.
    NaNs are preserved and excluded from the statistics.
    """
    x = np.asarray(img, dtype=np.float64)
    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)
    w = np.asarray(valid, dtype=np.float64)

    count = uniform_filter(w, size=window, mode="reflect")
    s = uniform_filter(x0, size=window, mode="reflect")
    s2 = uniform_filter(x0 * x0, size=window, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / count
        var = s2 / count - mean * mean
    var = np.clip(var, 0.0, None)

    # noise variance model for multiplicative speckle: sigma_n^2 = mean^2 / ENL
    noise_var = (mean * mean) / enl
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.clip((var - noise_var) / np.where(var > 0, var, np.inf), 0.0, 1.0)
    out = mean + k * (x0 - mean)
    out = out.astype(np.float32)
    out[~valid] = np.nan
    return out


def median_speckle(img: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    valid = np.isfinite(x)
    filled = np.where(valid, x, np.nanmedian(x) if valid.any() else 0.0)
    out = median_filter(filled, size=window, mode="reflect").astype(np.float32)
    out[~valid] = np.nan
    return out


def apply_speckle(img: np.ndarray, method: str = "lee", window: int = 5) -> np.ndarray:
    if method == "none":
        return np.asarray(img, dtype=np.float32)
    if method == "lee":
        return lee_filter(img, window=window)
    if method == "median":
        return median_speckle(img, window=window)
    raise ValueError(f"unknown speckle method {method!r}")
