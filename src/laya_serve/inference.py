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
"""

from __future__ import annotations

from typing import Any, Protocol

from .settings import Settings


class Backend(Protocol):
    """Minimal interface the API layer needs from any inference backend."""

    serving_model: str

    def predict(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Return ``{"answers": {...}, "usage": {...}}`` with raw backend answers."""
        ...


class FakeBackend:
    """Deterministic weight-free backend for tests and local development."""

    def __init__(self, serving_model: str = "laya-english") -> None:
        self.serving_model = serving_model
        self.calls: list[dict[str, Any]] = []

    def predict(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
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

    def predict(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        result = self._router.predict(state, questions)
        # Router adds a non-Jev ``routing`` key; compat shaping ignores it.
        return {
            "answers": result["answers"],
            "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }


def build_backend(settings: Settings) -> Backend:
    """Instantiate the backend selected by ``settings.backend``."""
    if settings.backend == "fake":
        return FakeBackend(serving_model=settings.serving_model)
    if settings.backend == "laya":
        return LayaBackend(settings)
    raise ValueError(f"unknown backend {settings.backend!r}; expected 'fake' or 'laya'")
