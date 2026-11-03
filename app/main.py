"""FastAPI service: submit a change-detection run, poll it, download its rasters."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .config import settings
from .models import ChangeRequest, ChangeResult, RunSummary
from .pipeline import run_change
from .store import make_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Sentinel-1 change detection",
    version="0.1.0",
    description=(
        "Pre/post SAR backscatter change on a GeoJSON AOI: flood extent, backscatter increase or any disturbance. "
        "Data from Copernicus Data Space (openEO), output as GeoTIFF + area statistics."
    ),
)
store = make_store(settings.database_url)

_OUTPUT_FILES = {"mask.tif": "image/tiff", "diff_db.tif": "image/tiff", "preview.png": "image/png", "result.json": "application/json"}


def _execute(run_id: str, req: ChangeRequest) -> ChangeResult:
    store.set_status(run_id, "running")
    try:
        result = run_change(req, run_id)
    except Exception as e:  # keep the API up; the error lands on the run
        logging.getLogger(__name__).exception("run %s failed", run_id)
        store.set_status(run_id, "failed", error=f"{e.__class__.__name__}: {e}")
        return store.get(run_id)  # type: ignore[return-value]
    result.outputs = {k: f"/runs/{run_id}/{Path(v).name}" for k, v in result.outputs.items()}
    store.complete(run_id, result)
    return result


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "default_backend": settings.default_backend, "persistence": "postgis" if settings.database_url else "memory"}


@app.post("/change", response_model=ChangeResult, status_code=202)
def submit_change(
    req: ChangeRequest,
    background: BackgroundTasks,
    wait: bool = Query(False, description="Run synchronously and return the completed result"),
) -> ChangeResult:
    run_id = uuid.uuid4().hex[:12]
    store.create(run_id, req)
    if wait or settings.sync:
        result = _execute(run_id, req)
        if result.status == "failed":
            raise HTTPException(status_code=422, detail=result.error)
        return result
    background.add_task(_execute, run_id, req)
    return store.get(run_id)  # type: ignore[return-value]


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[RunSummary]:
    return store.list(limit)


@app.get("/runs/{run_id}", response_model=ChangeResult)
def get_run(run_id: str) -> ChangeResult:
    r = store.get(run_id)
    if r is None:
        raise HTTPException(404, "unknown run")
    return r


@app.get("/runs/{run_id}/{filename}")
def get_run_file(run_id: str, filename: str):
    if filename not in _OUTPUT_FILES:
        raise HTTPException(404, f"no such output; choose one of {sorted(_OUTPUT_FILES)}")
    path = Path(settings.runs_dir) / run_id / filename
    if not path.exists():
        raise HTTPException(404, "output not available (run not finished or failed)")
    return FileResponse(path, media_type=_OUTPUT_FILES[filename], filename=f"{run_id}_{filename}")
