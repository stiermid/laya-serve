"""FastAPI application factory.

Endpoints (Jev-compatible):

* ``POST /v1/systemone`` — evaluate state + questions, return typed answers.
* ``GET /v1/models`` — list servable model names/aliases.
* ``GET /healthz`` — liveness probe (not part of the Jev API).
"""

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import compat
from .inference import Backend, build_backend
from .schemas import SystemOneRequest
from .service import evaluate, list_models
from .settings import Settings


def _require_auth(authorization: str | None, settings: Settings) -> None:
    """Enforce bearer auth iff ``settings.api_key`` is configured."""
    if settings.api_key is None:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


def create_app(
    settings: Settings | None = None,
    backend: Backend | None = None,
) -> FastAPI:
    settings = settings or Settings()
    backend = backend or build_backend(settings)

    app = FastAPI(title="laya-serve", version="0.1.0")
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
            content={"error": {"message": first.get("msg", "request failed validation"), "field": loc}},
        )

    @app.exception_handler(compat.CompatError)
    async def compat_handler(_: Request, exc: compat.CompatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": {"message": str(exc), "field": exc.field}},
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
