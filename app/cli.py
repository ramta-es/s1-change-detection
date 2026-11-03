"""Command line: run a request file without the API server, or validate a mask against a reference.

    python -m app.cli run examples/request_synthetic.json
    python -m app.cli run examples/request_cdse.json --out runs
    python -m app.cli validate runs/<id>/mask.tif reference_mask.tif
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .models import ChangeRequest
from .pipeline import run_change
from .validate import validate_mask


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="s1cd")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run change detection from a JSON request file")
    r.add_argument("request", type=Path)
    r.add_argument("--out", type=Path, default=Path("runs"))
    r.add_argument("--backend", choices=["cdse_openeo", "synthetic", "files"])
    r.add_argument("--run-id", default=None)

    v = sub.add_parser("validate", help="confusion matrix of a predicted mask against a reference mask")
    v.add_argument("pred", type=Path)
    v.add_argument("ref", type=Path)
    v.add_argument("--ref-positive", type=float, default=1.0, help="reference pixel value meaning 'changed'")

    a = p.parse_args(argv)
    if a.cmd == "run":
        req = ChangeRequest.model_validate(json.loads(a.request.read_text()))
        if a.backend:
            req = req.model_copy(update={"backend": a.backend})
        run_id = a.run_id or uuid.uuid4().hex[:12]
        result = run_change(req, run_id, runs_dir=a.out)
        print(result.model_dump_json(indent=2))
        return 0
    if a.cmd == "validate":
        report = validate_mask(a.pred, a.ref, ref_positive=a.ref_positive)
        print(json.dumps(report, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
