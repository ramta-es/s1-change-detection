"""Run persistence: in-memory by default, PostGIS when S1CD_DATABASE_URL is set."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any, Protocol

from .models import ChangeRequest, ChangeResult, RunSummary


class RunStore(Protocol):
    def create(self, run_id: str, req: ChangeRequest) -> None: ...
    def set_status(self, run_id: str, status: str, error: str | None = None) -> None: ...
    def complete(self, run_id: str, result: ChangeResult) -> None: ...
    def get(self, run_id: str) -> ChangeResult | None: ...
    def list(self, limit: int = 50) -> list[RunSummary]: ...


class MemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, run_id: str, req: ChangeRequest) -> None:
        with self._lock:
            self._runs[run_id] = {
                "created_at": datetime.now(UTC).isoformat(),
                "request": req,
                "result": ChangeResult(run_id=run_id, status="queued"),
            }

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            r = self._runs[run_id]["result"]
            self._runs[run_id]["result"] = r.model_copy(update={"status": status, "error": error})

    def complete(self, run_id: str, result: ChangeResult) -> None:
        with self._lock:
            self._runs[run_id]["result"] = result

    def get(self, run_id: str) -> ChangeResult | None:
        with self._lock:
            r = self._runs.get(run_id)
            return r["result"] if r else None

    def list(self, limit: int = 50) -> list[RunSummary]:
        with self._lock:
            items = sorted(self._runs.items(), key=lambda kv: kv[1]["created_at"], reverse=True)[:limit]
            return [RunSummary(run_id=k, status=v["result"].status, created_at=v["created_at"]) for k, v in items]


_DDL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    status      TEXT NOT NULL,
    error       TEXT,
    request     JSONB NOT NULL,
    result      JSONB,
    aoi         GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS runs_aoi_gix ON runs USING GIST (aoi);
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs (created_at DESC);
"""


class PostgisRunStore:
    """Minimal PostGIS store. Requires `pip install psycopg[binary]` and a PostGIS database."""

    def __init__(self, dsn: str) -> None:
        import psycopg  # noqa: F401  (import error surfaces at startup, not first request)

        self._dsn = dsn
        with self._conn() as c:
            c.execute(_DDL)
            c.commit()

    def _conn(self):
        import psycopg

        return psycopg.connect(self._dsn)

    def create(self, run_id: str, req: ChangeRequest) -> None:
        from shapely.geometry import mapping

        from .processing.geometry import parse_aoi, union_geom

        geom = union_geom(parse_aoi(req.aoi))
        if geom.geom_type == "Polygon":
            from shapely.geometry import MultiPolygon

            geom = MultiPolygon([geom])
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id, status, request, aoi) VALUES (%s, 'queued', %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (run_id, req.model_dump_json(), json.dumps(mapping(geom))),
            )
            c.commit()

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status=%s, error=%s WHERE run_id=%s", (status, error, run_id))
            c.commit()

    def complete(self, run_id: str, result: ChangeResult) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status='completed', result=%s WHERE run_id=%s", (result.model_dump_json(), run_id))
            c.commit()

    def get(self, run_id: str) -> ChangeResult | None:
        with self._conn() as c:
            row = c.execute("SELECT status, error, result FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if not row:
            return None
        status, error, result = row
        if result:
            return ChangeResult.model_validate(result)
        return ChangeResult(run_id=run_id, status=status, error=error)

    def list(self, limit: int = 50) -> list[RunSummary]:
        with self._conn() as c:
            rows = c.execute("SELECT run_id, status, created_at FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
        return [RunSummary(run_id=r[0], status=r[1], created_at=r[2].isoformat()) for r in rows]


def make_store(database_url: str | None) -> RunStore:
    if database_url:
        return PostgisRunStore(database_url)
    return MemoryRunStore()
