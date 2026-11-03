import numpy as np
import pytest

from app.backends.synthetic import make_pair
from app.processing.change import detect_change, otsu_threshold, to_db
from app.processing.speckle import lee_filter


def _scores(mask: np.ndarray, truth: np.ndarray):
    valid = mask != 255
    pred = mask == 1
    tp = np.sum(pred & truth & valid)
    fp = np.sum(pred & ~truth & valid)
    fn = np.sum(~pred & truth & valid)
    return tp / (tp + fp), tp / (tp + fn)


def test_flood_detected_with_otsu_and_lake_ignored(small_grid):
    pre, post, truth = make_pair(small_grid, seed=1)
    out = detect_change(pre, post, change_type="flood")
    precision, recall = _scores(out.mask, truth.flood)
    assert out.threshold_source == "otsu"
    assert out.threshold_db < -2.0
    assert precision > 0.97 and recall > 0.95
    # permanent water is dark on both dates: never "newly flooded"
    assert np.sum((out.mask == 1) & truth.lake) == 0
    # a backscatter increase is not a flood
    assert np.sum((out.mask == 1) & truth.bright) == 0


def test_increase_type_flags_bright_patch_only(small_grid):
    pre, post, truth = make_pair(small_grid, seed=2)
    out = detect_change(pre, post, change_type="increase")
    precision, recall = _scores(out.mask, truth.bright)
    assert precision > 0.9 and recall > 0.9
    assert np.sum((out.mask == 1) & truth.flood) == 0


def test_any_type_flags_both(small_grid):
    pre, post, truth = make_pair(small_grid, seed=3)
    out = detect_change(pre, post, change_type="any")
    both = truth.flood | truth.bright
    precision, recall = _scores(out.mask, both)
    assert precision > 0.9 and recall > 0.9


def test_user_threshold_is_respected(small_grid):
    pre, post, truth = make_pair(small_grid, seed=4)
    out = detect_change(pre, post, change_type="flood", threshold_db=-5.0)
    assert out.threshold_source == "user"
    assert out.threshold_db == -5.0
    _, recall = _scores(out.mask, truth.flood)
    assert recall > 0.9


def test_no_change_falls_back_and_flags_nothing(small_grid):
    pre, _, _ = make_pair(small_grid, seed=5)
    rng = np.random.default_rng(0)
    # same scene with fresh speckle only: no real change anywhere
    post = (pre * rng.gamma(4.4, 1 / 4.4, size=pre.shape)).astype(np.float32)
    out = detect_change(pre, post, change_type="flood")
    assert out.threshold_source == "fallback"
    assert any("fallback" in w for w in out.warnings)
    assert np.mean(out.mask[out.mask != 255] == 1) < 0.01


def test_min_mapping_unit_removes_specks(small_grid):
    pre, post, _truth = make_pair(small_grid, seed=6)
    # add a 2x2 isolated dark speck far from the flood
    post = post.copy()
    post[5:7, 5:7] = 1e-3
    out_mmu = detect_change(pre, post, change_type="flood", threshold_db=-3.0, min_mapping_unit_px=8)
    out_raw = detect_change(pre, post, change_type="flood", threshold_db=-3.0, min_mapping_unit_px=0, speckle_method="none")
    assert out_raw.mask[5:7, 5:7].sum() > 0
    assert out_mmu.mask[3:9, 3:9].sum() == 0


def test_exclude_mask_blocks_pixels(small_grid):
    pre, post, truth = make_pair(small_grid, seed=7)
    out = detect_change(pre, post, change_type="flood", threshold_db=-3.0, exclude_mask=truth.flood)
    assert np.sum((out.mask == 1) & truth.flood) == 0
    assert np.all(out.mask[truth.flood] == 255)


def test_nodata_propagates():
    pre = np.full((50, 50), 0.1, dtype=np.float32)
    post = pre.copy()
    pre[:5, :] = np.nan
    post[:, :5] = 0.0  # zeros are nodata too
    out = detect_change(pre, post, change_type="flood", threshold_db=-3.0, speckle_method="none", min_mapping_unit_px=0)
    assert np.all(out.mask[:5, :] == 255)
    assert np.all(out.mask[:, :5] == 255)
    assert np.all(out.mask[5:, 5:] == 0)


def test_to_db_and_lee_preserve_nan():
    x = np.array([[0.1, np.nan], [0.01, 0.0]], dtype=np.float32)
    db = to_db(x)
    assert np.isclose(db[0, 0], -10.0) and np.isnan(db[0, 1]) and np.isclose(db[1, 0], -20.0) and np.isnan(db[1, 1])
    f = lee_filter(np.where(np.isnan(x), np.nan, x), window=3)
    assert np.isnan(f[0, 1]) and np.isfinite(f[0, 0])


def test_otsu_separates_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0, 0.5, 5000), rng.normal(-8, 0.5, 1000)])
    t = otsu_threshold(v)
    assert -7 < t < -1


def test_otsu_rejects_tiny_sample():
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros(10))
