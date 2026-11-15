# Sentinel-1 change detection as a service

Pre/post SAR backscatter change on any GeoJSON AOI, delivered as a Dockerized FastAPI service:
flood extent, backscatter increase (debris, new structures) or generic ground disturbance.
Data comes from [Copernicus Data Space](https://dataspace.copernicus.eu) through openEO; outputs are
GeoTIFF masks, a dB difference raster, a PNG preview and area statistics per input polygon.

```
POST /change  {AOI, pre window, post window}  ->  run_id
GET  /runs/{id}                               ->  changed_area_km2, changed_fraction, per-polygon stats
GET  /runs/{id}/mask.tif | diff_db.tif | preview.png
```

![synthetic demo preview](docs/preview_synthetic.png)

*Synthetic demo: red = newly flooded, the dark lake (permanent water) is correctly ignored, the bright
patch is not a flood. Outside-AOI pixels are dimmed.*

## Why this exists

Most remote-sensing change detection stops at a notebook. This repo is the production shape of the
same work: a typed API, a persistent run log (PostGIS), a reproducible process graph, tests that run
without credentials, and a validation tool for checking the output against reference data. It is the
template behind the fixed-scope "Sentinel-1 change-detection pipeline" package I deliver to clients.

## Two-minute demo (no credentials)

```bash
pip install -e ".[dev]"
make demo                      # synthetic Sentinel-1-like scenes on a real AOI near Valencia
open runs/demo/preview.png
pytest -q                      # 32 tests, a few seconds
```

Or through the API:

```bash
S1CD_DEFAULT_BACKEND=synthetic uvicorn app.main:app --port 8000
curl -s -X POST "localhost:8000/change?wait=true" -H "content-type: application/json" \
     -d @examples/request_synthetic.json | jq .
```

## Real run on Copernicus Data Space

1. Create a free CDSE account and either register an OIDC client (`S1CD_CDSE_CLIENT_ID/SECRET` in
   `.env`) or leave both empty and complete the device-code login the first time.
2. `docker compose up --build` (API + PostGIS), then:

```bash
curl -s -X POST "localhost:8000/change" -H "content-type: application/json" -d @examples/request_cdse.json
# {"run_id":"7c1e...","status":"queued",...}
curl -s localhost:8000/runs/7c1e...            # poll until "completed"
curl -s localhost:8000/runs/7c1e.../mask.tif -o mask.tif
```

`examples/request_cdse.json` is the October 2024 DANA flood south of Valencia (pre 5–28 Oct, post
30 Oct–12 Nov, descending orbit, VV). A 200 km² AOI at 20 m runs in a few minutes on the free tier.

## Request

```jsonc
{
  "aoi": { "type": "FeatureCollection", "features": [ /* Polygons, WGS84; stats are reported per feature */ ] },
  "pre":  { "start": "2024-10-05", "end": "2024-10-28" },
  "post": { "start": "2024-10-30", "end": "2024-11-12" },
  "change_type": "flood",          // flood (decrease) | increase | any (|diff|)
  "polarization": "VV",            // VV for open water; VH is sometimes better for vegetation
  "orbit_state": "DESCENDING",     // keep pre and post on the same viewing geometry
  "relative_orbit": null,          // strongest geometry control, if you know the track
  "threshold_db": null,            // null = Otsu on the difference image, with a sanity floor
  "speckle": { "method": "lee", "window": 5 },
  "min_mapping_unit_px": 8,
  "resolution_m": 20,
  "backend": "cdse_openeo"         // cdse_openeo | files | synthetic
}
```

The `files` backend takes two GeoTIFFs you already have (`backend_options.pre_path/post_path`, linear
power or dB) so the same pipeline runs on SNAP or ASF output.

## Method

1. **Composites, not single scenes.** One openEO batch job computes the temporal median of
   backscatter over each window plus the number of valid observations per pixel. Medians over 2–4
   scenes remove more speckle and weather than any spatial filter; the counts are returned so a
   single-scene window is visible in the result (`n_pre_scenes`, `n_post_scenes`, warnings).
2. **Same geometry on both dates.** `orbit_state` or `relative_orbit` filters keep incidence angle and
   look direction constant; mixing ascending and descending passes creates false change on every
   slope and building.
3. **Radiometric terrain correction.** `sigma0-ellipsoid` by default; switch to `gamma0-terrain` in
   relief (`S1CD_CDSE_BACKSCATTER_COEFFICIENT`).
4. **Speckle filter** (Lee, ENL 4.4, or median) on linear power, then dB.
5. **Difference and threshold.** `diff = post − pre` in dB. Otsu on the valid difference pixels; for
   floods the threshold must be below −1.5 dB or the run falls back to −3 dB and says so
   (`threshold_source`), because Otsu on an image with no real flood returns noise.
6. **Masks.** Pixels darker than −20 dB in the pre image are water already and never "newly flooded";
   ESA WorldCover class 80 (permanent water) is excluded when available. Morphological opening and a
   minimum mapping unit remove speckle residue.
7. **Statistics** on an exact UTM grid: area in km², fraction of valid area, per input polygon.

## Validation

```bash
python -m app.cli validate runs/<id>/mask.tif reference_mask.tif --ref-positive 1
```

Reprojects the reference (e.g. a rasterised Copernicus EMS rapid-mapping product) onto the
prediction grid and reports TP/FP/FN/TN, precision, recall, F1, IoU, kappa and the predicted/reference
area ratio. Delivered client work ships with this report on at least two reference events.

## Known limitations

- **Layover and shadow** are not masked (no DEM in this pipeline yet). In mountains, use
  `gamma0-terrain` and expect false change on steep slopes.
- **Wind-roughened water** looks like land in SAR; a windy post-event scene underestimates flood extent.
  Composites over several scenes reduce this but do not remove it.
- **Flooded vegetation and urban flooding** show as a backscatter *increase* (double bounce), not a
  decrease. Run `change_type: "increase"` as a second pass if the AOI has crops or built-up areas.
- **Otsu assumes two populations.** Very small floods relative to the AOI need an explicit
  `threshold_db`.
- Property names used for collection filtering (`sat:orbit_state`, `relativeOrbitNumber`,
  `sar:instrument_mode`) should be checked against `describe_collection("SENTINEL1_GRD")` if a filter
  seems ignored; CDSE metadata names have changed before.

## Layout

```
app/
  main.py            FastAPI routes
  pipeline.py        request -> grid -> backend -> change -> stats -> files
  models.py          Pydantic request/response
  store.py           in-memory or PostGIS run store
  cli.py             run / validate without the server
  validate.py        confusion matrix against a reference mask
  backends/          cdse_openeo (openEO batch job), files (local GeoTIFFs), synthetic (tests/demo)
  processing/        geometry, speckle, change, stats, raster_io
tests/               32 tests; the openEO backend is tested against a fake connection and its calls are bound against the real client signatures
examples/            Valencia AOI, CDSE and synthetic requests
```

## Roadmap

- DEM-based layover/shadow mask (Copernicus DEM via openEO)
- Scheduled monitoring: re-run an AOI after every new acquisition and emit change events
- Sentinel-2 cloud-free composite as an optional optical cross-check
- Per-run cost and timing in the PostGIS log

## License

MIT
