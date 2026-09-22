"""Application service: orchestrates validation, inference, and compat shaping."""

from __future__ import annotations

from typing import Any

from . import compat
from .inference import Backend
from .schemas import SystemOneRequest
from .settings import Settings


def evaluate(request: SystemOneRequest, backend: Backend, settings: Settings) -> dict[str, Any]:
    """Run one SystemOne request and return a Jev-shaped response dict."""
    serving_model = compat.resolve_model(
        request.model, backend.serving_model, settings.extra_models_set
    )
    questions = request.model_dump(mode="json", exclude_none=False)["questions"]
    raw = backend.predict(request.state, questions)
    return compat.shape_response(
        questions,
        raw["answers"],
        raw.get("usage", {}),
        serving_model,
    )


def list_models(settings: Settings, backend: Backend) -> dict[str, Any]:
    """Return the models listing: the serving checkpoint plus Jev aliases."""
    from .schemas import ModelsResponse

    cards = [
        {
            "name": backend.serving_model,
            "description": "Laya decision model served with a Jev-compatible API.",
            "release_date": "unknown",
        }
    ]
    for alias in sorted(compat.JEV_KNOWN_MODELS | settings.extra_models_set):
        if alias == backend.serving_model:
            continue
        cards.append(
            {
                "name": alias,
                "description": f"Alias resolving to {backend.serving_model}.",
                "release_date": "unknown",
            }
        )
    return ModelsResponse.model_validate({"models": cards}).model_dump(mode="json")
