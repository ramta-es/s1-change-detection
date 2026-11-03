import numpy as np
import pytest
from pyproj import CRS

from app.processing.geometry import (
    geodesic_area_km2,
    make_grid,
    parse_aoi,
    pixel_area_km2,
    union_geom,
    utm_crs_for,
)
from app.processing.stats import area_stats, per_polygon_stats, rasterize_features

SQUARE = {"type": "Polygon", "coordinates": [[[34.75, 32.05], [34.85, 32.05], [34.85, 32.15], [34.75, 32.15], [34.75, 32.05]]]}


def test_parse_aoi_variants():
    assert len(parse_aoi(SQUARE)) == 1
    assert parse_aoi({"type": "Feature", "id": "a", "geometry": SQUARE})[0].feature_id == "a"
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"id": "x"}, "geometry": SQUARE}]}
    assert parse_aoi(fc)[0].feature_id == "x"
    with pytest.raises(ValueError):
        parse_aoi({"type": "Point", "coordinates": [0, 0]})
    with pytest.raises(ValueError):
        parse_aoi({"type": "Polygon", "coordinates": [[[400, 0], [401, 0], [401, 1], [400, 0]]]})


def test_geodesic_area_of_tenth_degree_square_near_tel_aviv():
    # 0.1° lat x 0.1° lon at 32.1°N: ~11.1 km x ~9.4 km
    a = geodesic_area_km2(union_geom(parse_aoi(SQUARE)))
    assert 100 < a < 110


def test_utm_zone_selection():
    assert utm_crs_for(union_geom(parse_aoi(SQUARE))) == CRS.from_epsg(32636)  # Israel: zone 36N
    south = {"type": "Polygon", "coordinates": [[[-70.0, -33.5], [-69.9, -33.5], [-69.9, -33.4], [-70.0, -33.4], [-70.0, -33.5]]]}
    assert utm_crs_for(union_geom(parse_aoi(south))) == CRS.from_epsg(32719)  # Santiago: zone 19S


def test_grid_snaps_to_resolution_and_covers_aoi():
    geom = union_geom(parse_aoi(SQUARE))
    grid = make_grid(geom, utm_crs_for(geom), 20)
    assert grid.transform.a == 20 and grid.transform.e == -20
    assert grid.transform.c % 20 == 0 and grid.transform.f % 20 == 0
    assert grid.width > 400 and grid.height > 500  # ~9.4 km / 20 m, ~11.1 km / 20 m
    px = pixel_area_km2(grid)
    assert px.shape == grid.shape and np.isclose(px[0, 0], 0.0004)


def test_geographic_pixel_area_uses_latitude():
    from rasterio.transform import from_origin

    from app.processing.geometry import Grid

    g = Grid(crs=CRS.from_epsg(4326), transform=from_origin(0, 60, 0.001, 0.001), width=10, height=10, resolution_m=100)
    px = pixel_area_km2(g)
    # at 60°N a 0.001° x 0.001° pixel is ~111.3 m x 55.7 m
    assert 0.0060 < px[0, 0] < 0.0064


def test_rasterize_and_stats(aoi_fc):
    feats = parse_aoi(aoi_fc)
    geom = union_geom(feats)
    grid = make_grid(geom, utm_crs_for(geom), 100)
    labels = rasterize_features(feats, grid)
    assert set(np.unique(labels)) == {0, 1, 2}
    mask = np.zeros(grid.shape, dtype=np.uint8)
    mask[labels == 1] = 1  # everything in polygon 1 changed
    rows_in_poly2 = np.where((labels == 2).any(axis=1))[0]
    mask[rows_in_poly2[:3], :] = 255  # a strip of nodata inside polygon 2
    inside = labels > 0
    tot = area_stats(mask, inside, grid)
    assert tot.aoi_area_km2 > tot.valid_area_km2 > tot.changed_area_km2 > 0
    per = dict(per_polygon_stats(mask, labels, feats, grid))
    assert per["horta-sud"].changed_fraction == pytest.approx(1.0)
    assert per["albufera-north"].changed_fraction == 0.0
    # polygon areas from the raster should be within 3 % of the geodesic area
    assert per["horta-sud"].aoi_area_km2 == pytest.approx(geodesic_area_km2(feats[0].geom), rel=0.03)
