"""Inference backend abstraction.

The HTTP layer depends only on the :class:`Backend` protocol, never on
``torch``/``laya`` directly, so the server (and its tests) import and run
without model weights installed:

* :class:`FakeBackend` — deterministic, weight-free answers. Uniform
  distributions for choice/score, ``0.5`` for noul. Used for tests and
  local API development.
* :class:`LayaBackend` — wraps ``laya.Router``. Import of ``laya`` (and
  therefore ``torch``) happens lazily inside the constructor, so merely
  importing this module stays cheap.

Context budgets (Jev: 64k total, 32k state+longest-question) are enforced
pre-inference from :meth:`Backend.count`, so oversize requests fail fast
with ``422`` instead of being silently truncated downstream. Enforcement
uses the deterministic weight-free word count on every backend so the
``200``/``422`` boundary never depends on which checkpoints happen to be
resident; per-request ``usage.input_tokens`` still comes from the Router.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .settings import Settings

# Jev token budgets per request (see ``docs.typesafe.ai/models``).
JEV_MAX_TOTAL_TOKENS = 65_536
JEV_MAX_STATE_QUESTION_TOKENS = 32_768


def render_text(value: Any) -> str:
    """Render one value as text, mirroring ``laya.common.render_criterion``."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_question_text(qdef: dict[str, Any]) -> str:
    """Rendered size basis of one question: instructions + criteria descriptions."""
    parts = [render_text(qdef.get("instructions", ""))]
    criteria = qdef.get("criteria")
    if isinstance(criteria, dict):
        parts.extend(render_text(v) for v in criteria.values() if v is not None)
    elif isinstance(criteria, list):
        parts.extend(render_text(level) for level in criteria)
    return " ".join(part for part in parts if part)


def count_request(state: Any, questions: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Weight-free (total, state+longest-question) word count of a request."""
    state_n = len(render_text(state).split())
    per_question = [len(render_question_text(q).split()) for q in questions.values()]
    longest = max(per_question, default=0)
    return state_n + sum(per_question), state_n + longest


class OverloadedError(Exception):
    """Transient backend overload; handled as a Jev-shaped ``529``.

    Backends raise this when inference could succeed on retry after a
    short delay (GPU OOM under burst load, evicted checkpoint reload,
    upstream timeout). Clients retry with backoff, honoring
    ``Retry-After``.
    """

    pass


# Substrings (lowercased) marking a generic exception as transient
# overload rather than a hard internal failure. Matched against
# ``str(exc)`` and the exception type name, so torch OOMs
# (``torch.cuda.OutOfMemoryError`` / ``RuntimeError: CUDA out of
# memory``) map to ``529`` without importing torch here.
_OVERLOAD_MARKERS = frozenset(
    {
        "out of memory",
        "outofmemory",
        "cuda oom",
        "overloaded",
        "temporarily unavailable",
        "temporarily overloaded",
        "capacity",
        "backpressure",
        "too many requests",
        "timed out",
        "timeout",
        "busy",
    }
)


def is_overload(exc: BaseException) -> bool:
    """Return whether ``exc`` looks like transient overload (retryable)."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _OVERLOAD_MARKERS)


class Backend(Protocol):
    """Minimal interface the API layer needs from any inference backend."""

    serving_model: str
    # Token budgets enforced pre-inference (Jev: 64k total, 32k
    # state+longest-question). Backends that predate this interface simply
    # omit them and skip enforcement.
    max_total_tokens: int
    max_state_question_tokens: int

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Return ``{"answers": {...}, "usage": {...}}`` with raw backend answers."""
        ...

    def count(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[int, int]:
        """Return ``(total, state_plus_longest)`` token estimates for budgets."""
        ...


class FakeBackend:
    """Deterministic weight-free backend for tests and local development."""

    max_total_tokens = JEV_MAX_TOTAL_TOKENS
    max_state_question_tokens = JEV_MAX_STATE_QUESTION_TOKENS

    def __init__(self, serving_model: str = "laya-english") -> None:
        self.serving_model = serving_model
        self.calls: list[dict[str, Any]] = []

    def count(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[int, int]:
        return count_request(state, questions)

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        answers: dict[str, Any] = {}
        for qid, qdef in questions.items():
            qtype = qdef["type"]
            if qtype == "noul":
                answers[qid] = {"type": "noul", "noul": 0.5}
            elif qtype == "choice":
                options = list(qdef["criteria"])
                prob = 1.0 / len(options)
                answers[qid] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {opt: prob for opt in options},
                    "confidence": 0.0 if len(options) > 1 else 1.0,
                }
            elif qtype == "score":
                levels = list(qdef["criteria"])
                prob = 1.0 / len(levels)
                answers[qid] = {
                    "type": "score",
                    "score": (len(levels) - 1) / 2.0,
                    "legend": {str(i): level for i, level in enumerate(levels)},
                    "probabilities": {str(i): prob for i in range(len(levels))},
                    "confidence": 0.0,
                }
            else:  # pragma: no cover - validated upstream
                raise ValueError(f"unknown question type {qtype!r}")
        return {"answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


class LayaBackend:
    """Production backend wrapping ``laya.Router`` (lazy import)."""

    max_total_tokens = JEV_MAX_TOTAL_TOKENS
    max_state_question_tokens = JEV_MAX_STATE_QUESTION_TOKENS

    def __init__(self, settings: Settings) -> None:
        try:
            from laya import Router
        except ImportError as exc:
            raise RuntimeError(
                "laya is not installed; install the 'inference' extra "
                "(pip install 'laya-serve[inference]') or use backend='fake'"
            ) from exc
        self.serving_model = settings.serving_model
        self._router = Router(
            device=settings.device,
            max_loaded=settings.max_loaded,
            preload=settings.preload,
        )

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        try:
            result = self._router.predict(state, questions)
        except OverloadedError:
            raise
        except Exception as exc:
            if is_overload(exc):
                raise OverloadedError(f"backend overloaded: {exc}") from exc
            raise
        # Router adds a non-Jev ``routing`` key; compat shaping ignores it.
        return {
            "answers": result["answers"],
            "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }

    def count(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[int, int]:
        """Estimate request size with the deterministic weight-free word count.

        Deliberately independent of resident checkpoint tokenizers so the
        ``200``/``422`` boundary is stable across cold starts, evictions,
        and ``max_loaded`` rotation. Word counts underestimate true tokens,
        so enforcement is lenient (fail-open); precise
        ``usage.input_tokens`` still comes from the Router in
        :meth:`predict`.
        """
        return count_request(state, questions)


def build_backend(settings: Settings) -> Backend:
    """Instantiate the backend selected by ``settings.backend``."""
    if settings.backend == "fake":
        return FakeBackend(serving_model=settings.serving_model)
    if settings.backend == "laya":
        return LayaBackend(settings)
    raise ValueError(f"unknown backend {settings.backend!r}; expected 'fake' or 'laya'")
