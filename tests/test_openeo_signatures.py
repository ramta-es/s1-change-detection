"""Guard against drift between our calls and the installed openeo client.

The fake connection in test_openeo_backend.py accepts any kwargs, so a misspelled or removed parameter
would only fail on a real run. Here every call made by build_cube/build_worldcover is checked against the
real DataCube / Connection signatures, no network needed.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest
from openeo.rest.connection import Connection
from openeo.rest.datacube import DataCube

from app.backends.base import StackRequest
from app.backends.openeo_cdse import CdseOpenEOBackend

METHOD_OWNER = {"load_collection": Connection}


class RecordingCube:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return RecordingCube(self.calls)

        return _call


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def load_collection(self, *args, **kwargs):
        self.calls.append(("load_collection", args, kwargs))
        return RecordingCube(self.calls)


def _bind(owner, name, args, kwargs):
    fn = getattr(owner, name)
    sig = inspect.signature(fn)
    # bind with a dummy self; raises TypeError on unknown or missing parameters
    sig.bind(None, *args, **kwargs)


@pytest.fixture
def calls(small_grid):
    conn = RecordingConnection()
    be = CdseOpenEOBackend(connection_factory=lambda: conn)
    req = StackRequest(
        grid=small_grid,
        bbox_wgs84=(-0.48, 39.30, -0.30, 39.46),
        pre=(date(2024, 10, 5), date(2024, 10, 28)),
        post=(date(2024, 10, 30), date(2024, 11, 12)),
        orbit_state="DESCENDING",
        relative_orbit=110,
    )
    be.build_cube(conn, req)
    be.build_worldcover(conn, req)
    return conn.calls


def test_every_call_binds_to_the_real_client_signature(calls):
    assert calls, "no calls recorded"
    for name, args, kwargs in calls:
        owner = METHOD_OWNER.get(name, DataCube)
        assert hasattr(owner, name), f"{owner.__name__} has no method {name!r}"
        try:
            _bind(owner, name, args, kwargs)
        except TypeError as e:
            pytest.fail(f"{owner.__name__}.{name}{args}{kwargs}: {e}")


def test_resample_methods_are_documented_values(calls):
    import openeo.processes as processes

    methods = {kw["method"] for n, _, kw in calls if n == "resample_spatial"}
    spec_doc = inspect.getdoc(processes.resample_spatial) or ""  # bundled openEO process spec
    for m in methods:
        assert f"`{m}`" in spec_doc, f"resample method {m!r} not in the resample_spatial process spec"
