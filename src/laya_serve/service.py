"""Application service: orchestrates validation, inference, and compat shaping."""

from __future__ import annotations

import logging
import re
from typing import Any

from . import compat
from .inference import Backend
from .schemas import SystemOneRequest
from .settings import Settings

logger = logging.getLogger(__name__)


def enforce_budgets(state: Any, questions: dict[str, dict[str, Any]], backend: Backend) -> None:
    """Fail fast with ``422`` when the request exceeds the backend's token budgets.

    Compares ``backend.count(state, questions)`` against the backend's
    ``max_total_tokens`` / ``max_state_question_tokens`` (Jev: 64k total,
    32k state+longest-question). Backends that predate the budget
    interface skip enforcement. A broken counter fails open (warn + skip)
    so a counting bug can never turn a small request into a ``500``.
    """
    count = getattr(backend, "count", None)
    max_total = getattr(backend, "max_total_tokens", None)
    max_longest = getattr(backend, "max_state_question_tokens", None)
    if not callable(count) or max_total is None or max_longest is None:
        return
    try:
        total, state_longest = (int(n) for n in count(state, questions))
    except Exception:
        logger.warning("budget count failed; skipping enforcement", exc_info=True)
        return
    if state_longest > max_longest:
        raise compat.CompatError(
            f"context budget exceeded: state plus longest question is ~{state_longest} "
            f"tokens (limit {max_longest})",
            field="questions",
        )
    if total > max_total:
        raise compat.CompatError(
            f"context budget exceeded: state plus all questions is ~{total} tokens "
            f"(limit {max_total})",
            field="questions",
        )


def _as_budget_error(exc: ValueError) -> Exception:
    """Map head-budget overflow to ``CompatError`` (``422``); pass the rest through."""
    message = str(exc)
    lowered = message.lower()
    if "head_max_len" not in lowered and "options exceed" not in lowered:
        return exc
    match = re.search(r"question '([^']+)'", message) or re.search(r'question "([^"]+)"', message)
    field = f"questions.{match.group(1)}.criteria" if match else "questions"
    return compat.CompatError(message, field=field)


def evaluate(request: SystemOneRequest, backend: Backend, settings: Settings) -> dict[str, Any]:
    """Run one SystemOne request and return a Jev-shaped response dict."""
    serving_model = compat.resolve_model(
        request.model, backend.serving_model, settings.extra_models_set
    )
    questions = request.model_dump(mode="json", exclude_none=False)["questions"]
    enforce_budgets(request.state, questions, backend)
    try:
        raw = backend.predict(request.state, questions)
    except ValueError as exc:
        mapped = _as_budget_error(exc)
        if mapped is exc:
            raise
        raise mapped from exc
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
