"""Translate raw Laya backend payloads into the Jev wire format.

Known divergences between Laya's ``Agent.system_one`` output and Jev
(verified against ``laya/agent.py`` and ``docs.typesafe.ai/api.md``):

1. Laya attaches ``action: {act_probability}`` to every answer; Jev has
   no such field. Stripped here.
2. Laya returns ``confidence`` on ``noul`` answers; Jev ``noul`` answers
   carry only ``{type, noul}``. Stripped here.
3. Laya reports ``model: "laya-rl-agent"``; Jev echoes the resolved
   versioned model id. The caller supplies the id to report.
4. Laya always reports ``output_tokens: 0``; Jev reports a (free but
   nonzero) count. Preserved as-is and documented — this layer does not
   invent token counts.

All functions here are pure and backend-agnostic: they operate on plain
dicts, so they are unit-testable without model weights.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    StructuredText,
    SystemOneResponse,
    Usage,
)

# Model names this server accepts in the ``model`` request field out of the
# box. Anything else is a 422 (fail fast on typos, mirroring Jev's
# validation behaviour). Operators can extend this via settings.
JEV_KNOWN_MODELS = frozenset(
    {
        "jev-latest",
        "jev-preview",
        "jev-1.13.0",
        "jev-1.13",
        "laya",
        "laya-latest",
    }
)


class CompatError(ValueError):
    """Raised when a backend payload cannot be shaped into a Jev answer."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def resolve_model(
    requested: str, serving_model: str, extra_models: frozenset[str] = frozenset()
) -> str:
    """Validate the requested ``model`` and return the id to report.

    Known Jev names (plus operator-configured extras) resolve to
    ``serving_model`` — the checkpoint actually answering, e.g.
    ``"laya-english"``. Unknown names raise :class:`CompatError`.
    """
    if requested in JEV_KNOWN_MODELS or requested in extra_models or requested == serving_model:
        return serving_model
    raise CompatError(
        f"unknown model {requested!r}; expected a Jev model name or the serving model {serving_model!r}",
        field="model",
    )


def _clamp01(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompatError(f"expected a number for {field}, got {value!r}", field=field) from exc
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        raise CompatError(f"expected a finite number for {field}, got {value!r}", field=field)
    return min(1.0, max(0.0, number))


def shape_noul(question_id: str, raw: dict[str, Any]) -> NoulAnswer:
    """Shape a raw ``noul`` answer; drops Laya-only ``confidence``/``action``."""
    if "noul" not in raw:
        raise CompatError(
            f"noul answer {question_id!r} is missing 'noul'", field=f"answers.{question_id}"
        )
    return NoulAnswer(type="noul", noul=_clamp01(raw["noul"], f"answers.{question_id}.noul"))


def shape_choice(question_id: str, raw: dict[str, Any], options: list[str]) -> ChoiceAnswer:
    """Shape a raw ``choice`` answer; drops Laya-only ``action``."""
    choice = raw.get("choice")
    if choice not in options:
        raise CompatError(
            f"choice answer {question_id!r} selected {choice!r}, which is not one of {options!r}",
            field=f"answers.{question_id}.choice",
        )
    probs = raw.get("probabilities", {})
    if set(probs) != set(options):
        raise CompatError(
            f"choice answer {question_id!r} probabilities cover {sorted(probs)!r}, expected {options!r}",
            field=f"answers.{question_id}.probabilities",
        )
    return ChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={
            opt: _clamp01(probs[opt], f"answers.{question_id}.probabilities.{opt}")
            for opt in options
        },
        confidence=_clamp01(raw.get("confidence"), f"answers.{question_id}.confidence"),
    )


def shape_score(
    question_id: str,
    raw: dict[str, Any],
    levels: list[StructuredText],
) -> ScoreAnswer:
    """Shape a raw ``score`` answer with stringified level keys; drops ``action``."""
    expected = [str(i) for i in range(len(levels))]
    probs = raw.get("probabilities", {})
    if set(map(str, probs)) != set(expected):
        raise CompatError(
            f"score answer {question_id!r} probabilities cover {sorted(map(str, probs))!r}, "
            f"expected levels {expected!r}",
            field=f"answers.{question_id}.probabilities",
        )
    try:
        score = float(raw["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompatError(
            f"score answer {question_id!r} is missing a numeric 'score'",
            field=f"answers.{question_id}.score",
        ) from exc
    if not 0.0 <= score <= float(len(levels) - 1):
        raise CompatError(
            f"score answer {question_id!r} has score {score!r} outside [0, {len(levels) - 1}]",
            field=f"answers.{question_id}.score",
        )
    return ScoreAnswer(
        type="score",
        score=score,
        legend={str(i): level for i, level in enumerate(levels)},
        probabilities={
            str(i): _clamp01(
                probs[i] if i in probs else probs[str(i)],
                f"answers.{question_id}.probabilities.{i}",
            )
            for i in range(len(levels))
        },
        confidence=_clamp01(raw.get("confidence"), f"answers.{question_id}.confidence"),
    )


def shape_response(
    request_questions: dict[str, dict[str, Any]],
    raw_answers: dict[str, dict[str, Any]],
    usage: dict[str, Any],
    serving_model: str,
) -> dict[str, Any]:
    """Shape a full backend result into a Jev response dict (validated by caller).

    ``request_questions`` are the *validated* question dicts (with ``type``
    and ``criteria``); they define the expected answer shapes. ``raw_answers``
    is the backend's per-question output. Unknown/extra backend keys
    (``action``, ``routing``) are ignored.
    """
    if set(raw_answers) != set(request_questions):
        raise CompatError(
            f"backend answered {sorted(raw_answers)!r}, expected {sorted(request_questions)!r}",
            field="answers",
        )
    answers: dict[str, Any] = {}
    for qid, qdef in request_questions.items():
        raw = raw_answers[qid]
        qtype = qdef["type"]
        if qtype == "noul":
            answers[qid] = shape_noul(qid, raw)
        elif qtype == "choice":
            criteria = qdef["criteria"]
            options = list(criteria) if isinstance(criteria, dict) else list(criteria)
            answers[qid] = shape_choice(qid, raw, options)
        elif qtype == "score":
            answers[qid] = shape_score(qid, raw, list(qdef["criteria"]))
        else:  # pragma: no cover - schema validation rejects this first
            raise CompatError(f"unknown question type {qtype!r}", field=f"questions.{qid}.type")
    response = SystemOneResponse(
        model=serving_model,
        answers=answers,
        usage=Usage(
            input_tokens=max(0, int(usage.get("input_tokens", 0))),
            output_tokens=max(0, int(usage.get("output_tokens", 0))),
        ),
    )
    return response.model_dump(mode="json", exclude_none=False)
