"""Pre/post backscatter difference, thresholding and clean-up."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_opening, label

from .speckle import apply_speckle

# Below this (linear) power the pixel is treated as nodata: typical for border noise and true zeros.
_MIN_POWER = 1e-6
# Pixels already this dark in the PRE image are water/very smooth surfaces, not newly flooded.
PRE_WATER_DB = -20.0
# Otsu sanity floors: a flood threshold weaker than this means Otsu found noise, not a flood.
OTSU_MIN_DECREASE_DB = -1.5
FALLBACK_DECREASE_DB = -3.0
FALLBACK_INCREASE_DB = 3.0


@dataclass
class ChangeOutput:
    pre_db: np.ndarray
    post_db: np.ndarray
    diff_db: np.ndarray
    mask: np.ndarray  # uint8: 1 = changed, 0 = unchanged, 255 = nodata / masked
    threshold_db: float
    threshold_source: str  # user | otsu | fallback
    warnings: list[str] = field(default_factory=list)


def to_db(power: np.ndarray) -> np.ndarray:
    p = np.asarray(power, dtype=np.float32)
    out = np.full(p.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(p) & (p > _MIN_POWER)
    out[ok] = 10.0 * np.log10(p[ok])
    return out


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold on a 1-D sample. Returns the bin edge that maximises between-class variance."""
    v = values[np.isfinite(values)]
    if v.size < 100:
        raise ValueError("not enough valid pixels for Otsu")
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        raise ValueError("degenerate difference image")
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.where(w0 > 0, w0, np.nan)
    m1 = (np.cumsum((hist * centers)[::-1])[::-1] - hist * centers) / np.where(w1 > 0, w1, np.nan)
    between = w0 * w1 * (m0 - m1) ** 2
    between = np.where(np.isfinite(between), between, -np.inf)
    k = int(np.nanargmax(between))
    return float(edges[k + 1])


def detect_change(
    pre_power: np.ndarray,
    post_power: np.ndarray,
    change_type: str = "flood",
    threshold_db: float | None = None,
    speckle_method: str = "lee",
    speckle_window: int = 5,
    min_mapping_unit_px: int = 8,
    exclude_mask: np.ndarray | None = None,
) -> ChangeOutput:
    """Run the full change pipeline on two co-registered linear-power arrays (H, W).

    exclude_mask: boolean (H, W), True where pixels must never be flagged (permanent water, layover/shadow).
    """
    if pre_power.shape != post_power.shape:
        raise ValueError(f"pre/post shape mismatch: {pre_power.shape} vs {post_power.shape}")
    warnings: list[str] = []

    pre_f = apply_speckle(pre_power, speckle_method, speckle_window)
    post_f = apply_speckle(post_power, speckle_method, speckle_window)
    pre_db = to_db(pre_f)
    post_db = to_db(post_f)
    diff = post_db - pre_db
    valid = np.isfinite(diff)
    if exclude_mask is not None:
        valid &= ~exclude_mask.astype(bool)

    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid_fraction < 0.2:
        warnings.append(f"only {valid_fraction:.0%} of the grid has valid pre and post data")

    # --- threshold ------------------------------------------------------------
    source = "user"
    if threshold_db is None:
        try:
            t = otsu_threshold(diff[valid])
        except ValueError as e:
            warnings.append(f"Otsu failed ({e}); using fallback threshold")
            t = None
        if change_type == "flood":
            if t is None or t > OTSU_MIN_DECREASE_DB:
                if t is not None:
                    warnings.append(
                        f"Otsu threshold {t:+.2f} dB is weaker than {OTSU_MIN_DECREASE_DB} dB; "
                        f"likely no clear flood signal, using fallback {FALLBACK_DECREASE_DB} dB"
                    )
                t, source = FALLBACK_DECREASE_DB, "fallback"
            else:
                source = "otsu"
        elif change_type == "increase":
            if t is None or t < -OTSU_MIN_DECREASE_DB:
                t, source = FALLBACK_INCREASE_DB, "fallback"
            else:
                source = "otsu"
        else:  # any: Otsu on |diff|
            try:
                t = otsu_threshold(np.abs(diff[valid]))
                source = "otsu"
                if t < -OTSU_MIN_DECREASE_DB:
                    t, source = -FALLBACK_DECREASE_DB, "fallback"
            except ValueError:
                t, source = -FALLBACK_DECREASE_DB, "fallback"
        threshold_db = float(t)

    # --- classify -------------------------------------------------------------
    if change_type == "flood":
        changed = diff < threshold_db
        # newly flooded means it was NOT water before
        changed &= ~(pre_db < PRE_WATER_DB)
    elif change_type == "increase":
        changed = diff > threshold_db
    elif change_type == "any":
        changed = np.abs(diff) > abs(threshold_db)
    else:
        raise ValueError(f"unknown change_type {change_type!r}")
    changed &= valid

    # --- clean-up: opening + minimum mapping unit ---------------------------------
    if min_mapping_unit_px > 0 and changed.any():
        changed = binary_opening(changed, structure=np.ones((3, 3), dtype=bool))
        labels, n = label(changed, structure=np.ones((3, 3), dtype=int))
        if n:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= min_mapping_unit_px
            keep[0] = False
            changed = keep[labels]

    mask = np.full(diff.shape, 255, dtype=np.uint8)
    mask[valid] = 0
    mask[changed] = 1

    return ChangeOutput(
        pre_db=pre_db,
        post_db=post_db,
        diff_db=diff.astype(np.float32),
        mask=mask,
        threshold_db=float(threshold_db),
        threshold_source=source,
        warnings=warnings,
    )
