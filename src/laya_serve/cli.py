"""CLI entry point: ``laya-serve``."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .settings import Settings


def main() -> None:
    """Serve the app with uvicorn (host/port via UVIcorn env or defaults)."""
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
