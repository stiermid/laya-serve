"""CLI entry point: ``laya-serve``."""

from __future__ import annotations

from collections.abc import Sequence

import uvicorn

from . import __version__
from .app import create_app
from .settings import Settings


def main(argv: Sequence[str] | None = None) -> None:
    """Serve the app with uvicorn (host/port via flags, defaults ``0.0.0.0:8000``)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="laya-serve",
        description="Jev-compatible HTTP server for Laya decision models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    args = parser.parse_args(argv)

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
