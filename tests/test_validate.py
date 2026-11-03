import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS as RioCRS

from app.validate import confusion, validate_mask


def test_confusion_metrics():
    pred = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    ref = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=bool)
    valid = np.ones_like(pred)
    valid[0, 3] = False
    m = confusion(pred, ref, valid)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == m["recall"] == m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["iou"] == 0.5
    assert m["n_valid_px"] == 7


def test_validate_mask_reprojects_reference(small_grid, tmp_path):
    h, w = small_grid.shape
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[10:60, 10:60] = 1
    pred[0, :] = 255
    ref = np.zeros((h, w), dtype=np.uint8)
    ref[10:60, 10:70] = 1  # reference is wider: 50x60 vs predicted 50x50
    profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "crs": RioCRS.from_wkt(small_grid.crs.to_wkt()), "transform": small_grid.transform}
    with rasterio.open(tmp_path / "pred.tif", "w", dtype="uint8", nodata=255, **profile) as d:
        d.write(pred, 1)
    with rasterio.open(tmp_path / "ref.tif", "w", dtype="uint8", **profile) as d:
        d.write(ref, 1)
    rep = validate_mask(tmp_path / "pred.tif", tmp_path / "ref.tif")
    assert rep["tp"] == 2500 and rep["fp"] == 0 and rep["fn"] == 500
    assert rep["recall"] == pytest.approx(2500 / 3000, abs=1e-4) and rep["precision"] == 1.0
    assert rep["n_valid_px"] == h * w - w
