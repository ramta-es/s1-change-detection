"""Validation against a reference mask (e.g. Copernicus EMS rapid-mapping product rasterised, or hand-digitised)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.warp import Resampling, reproject


def confusion(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    p = pred[valid].astype(bool)
    r = ref[valid].astype(bool)
    tp = int(np.sum(p & r))
    fp = int(np.sum(p & ~r))
    fn = int(np.sum(~p & r))
    tn = int(np.sum(~p & ~r))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "n_valid_px": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "overall_accuracy": round(po, 4),
        "kappa": round(kappa, 4),
        "area_ratio_pred_over_ref": round((tp + fp) / (tp + fn), 4) if tp + fn else None,
    }


def validate_mask(pred_path: Path, ref_path: Path, ref_positive: float = 1.0) -> dict:
    """Reprojects the reference onto the prediction grid (nearest) and reports a confusion matrix.

    Prediction nodata (255) and reference nodata are excluded from the comparison.
    """
    with rasterio.open(pred_path) as ps:
        pred = ps.read(1)
        pred_nodata = ps.nodata if ps.nodata is not None else 255
        ref = np.full(pred.shape, np.nan, dtype=np.float32)
        with rasterio.open(ref_path) as rs:
            reproject(
                source=rasterio.band(rs, 1),
                destination=ref,
                src_transform=rs.transform,
                src_crs=rs.crs,
                src_nodata=rs.nodata,
                dst_transform=ps.transform,
                dst_crs=RioCRS.from_wkt(ps.crs.to_wkt()),
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    valid = (pred != pred_nodata) & np.isfinite(ref)
    report = confusion(pred == 1, ref == ref_positive, valid)
    report["pred"] = str(pred_path)
    report["ref"] = str(ref_path)
    return report
