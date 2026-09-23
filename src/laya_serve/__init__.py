"""laya-serve: Jev-compatible HTTP server for Laya decision models."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("laya-serve")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
