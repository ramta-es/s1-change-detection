import rasterio
from fastapi.testclient import TestClient

from app.main import app, store


def _client() -> TestClient:
    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_change_sync_roundtrip(request_json, _runs_dir):
    c = _client()
    r = c.post("/change", params={"wait": "true"}, json=request_json)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["backend"] == "synthetic"
    assert body["crs"] == "EPSG:32630"
    assert body["threshold_source"] == "otsu"
    assert 0.05 < body["changed_fraction"] < 0.3
    assert {p["feature_id"] for p in body["polygons"]} == {"horta-sud", "albufera-north"}
    assert body["outputs"]["mask_tif"] == f"/runs/{body['run_id']}/mask.tif"

    run_id = body["run_id"]
    assert c.get(f"/runs/{run_id}").json()["status"] == "completed"
    assert any(x["run_id"] == run_id for x in c.get("/runs").json())

    tif = c.get(f"/runs/{run_id}/mask.tif")
    assert tif.status_code == 200 and tif.headers["content-type"] == "image/tiff"
    with rasterio.open(_runs_dir / run_id / "mask.tif") as src:
        assert src.count == 1 and src.nodata == 255 and src.crs.to_epsg() == 32630
        assert src.overviews(1)  # overviews built
    png = c.get(f"/runs/{run_id}/preview.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_change_background_mode(request_json):
    c = _client()
    r = c.post("/change", json=request_json)
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    # TestClient runs background tasks before returning, so the run is already finished
    assert store.get(run_id).status == "completed"


def test_invalid_windows_rejected(request_json):
    bad = dict(request_json, post={"start": "2024-10-01", "end": "2024-10-02"})
    r = _client().post("/change", json=bad)
    assert r.status_code == 422
    assert "post window" in r.text


def test_unknown_run_and_file():
    c = _client()
    assert c.get("/runs/nope").status_code == 404
    assert c.get("/runs/nope/mask.tif").status_code == 404
    assert c.get("/runs/nope/../../etc/passwd").status_code == 404


def test_backend_failure_is_reported(request_json):
    bad = dict(request_json, backend="files", backend_options={"pre_path": "/no/such.tif", "post_path": "/no/such2.tif"})
    r = _client().post("/change", params={"wait": "true"}, json=bad)
    assert r.status_code == 422
    assert "FileNotFoundError" in r.json()["detail"]
