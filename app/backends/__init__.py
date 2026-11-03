from __future__ import annotations

from .base import Backend, BackscatterStack, StackRequest


def get_backend(name: str) -> Backend:
    if name == "synthetic":
        from .synthetic import SyntheticBackend

        return SyntheticBackend()
    if name == "files":
        from .files import FilesBackend

        return FilesBackend()
    if name == "cdse_openeo":
        from .openeo_cdse import CdseOpenEOBackend

        return CdseOpenEOBackend()
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["Backend", "BackscatterStack", "StackRequest", "get_backend"]
