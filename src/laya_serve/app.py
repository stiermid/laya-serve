"""FastAPI application factory.

Endpoints (Jev-compatible):

* ``POST /v1/systemone`` — evaluate state + questions, return typed answers.
* ``GET /v1/models`` — list servable model names/aliases.
* ``GET /healthz`` — liveness probe (not part of the Jev API).

Errors are always Jev-shaped (``{"error": {"message", "field"}}``):
``401`` bad key, ``422`` validation, ``529`` transient overload (with
``Retry-After``), ``500`` unexpected backend failure.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__ as _package_version
from . import compat
from .inference import Backend, OverloadedError, build_backend, is_overload
from .schemas import SystemOneRequest
from .service import evaluate, list_models
from .settings import Settings

logger = logging.getLogger(__name__)

# Jev overload status has no ``starlette.status`` constant (non-standard).
HTTP_529_OVERLOADED = 529


class AuthError(Exception):
    """Raised when bearer auth fails; handled as a Jev-shaped ``401``."""

    pass


def _require_auth(authorization: str | None, settings: Settings) -> None:
    """Enforce bearer auth iff ``settings.api_key`` is configured."""
    if settings.api_key is None:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise AuthError("Missing or invalid API key.")


def create_app(
    settings: Settings | None = None,
    backend: Backend | None = None,
) -> FastAPI:
    settings = settings or Settings()
    backend = backend or build_backend(settings)

    app = FastAPI(title="laya-serve", version=_package_version)
    app.state.settings = settings
    app.state.backend = backend

    def get_settings(request: Request) -> Settings:
        return request.app.state.settings

    def get_backend(request: Request) -> Backend:
        return request.app.state.backend

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ())) or None
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {"message": first.get("msg", "request failed validation"), "field": loc}
            },
        )

    @app.exception_handler(compat.CompatError)
    async def compat_handler(_: Request, exc: compat.CompatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": {"message": str(exc), "field": exc.field}},
        )

    @app.exception_handler(AuthError)
    async def auth_handler(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": {"message": str(exc), "field": None}},
        )

    @app.exception_handler(OverloadedError)
    async def overloaded_handler(_: Request, exc: OverloadedError) -> JSONResponse:
        logger.warning("backend overloaded: %s", exc)
        return JSONResponse(
            status_code=HTTP_529_OVERLOADED,
            content={"error": {"message": str(exc), "field": None}},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Preserve the status (404/405/…) but keep the Jev error body so
        # SDK clients never see FastAPI's ``{"detail": …}`` shape.
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "field": None}},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        # Raw backend bugs (torch errors, malformed payloads) must not
        # leak as ``500 text/plain``; shape them for SDK error parsing.
        # OOM-like messages are transient: report 529 so clients back off.
        if is_overload(exc):
            logger.warning("backend overloaded: %r", exc)
            return JSONResponse(
                status_code=HTTP_529_OVERLOADED,
                content={"error": {"message": f"backend overloaded: {exc}", "field": None}},
                headers={"Retry-After": "1"},
            )
        logger.exception("unhandled error processing request")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": "Internal server error.", "field": None}},
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(
        settings_dep: Annotated[Any, Depends(get_settings)],
        backend_dep: Annotated[Any, Depends(get_backend)],
    ) -> dict[str, Any]:
        return list_models(settings_dep, backend_dep)

    @app.post("/v1/systemone", status_code=status.HTTP_200_OK)
    def systemone(
        request: SystemOneRequest,
        settings_dep: Annotated[Any, Depends(get_settings)],
        backend_dep: Annotated[Any, Depends(get_backend)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        _require_auth(authorization, settings_dep)
        return evaluate(request, backend_dep, settings_dep)

    return app
