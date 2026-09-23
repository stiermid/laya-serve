"""Fixture for issue #2: context budgets are not enforced.

Jev enforces two token budgets per request (see ``docs.typesafe.ai/models``):

* ``max_total_tokens`` — state plus ALL questions combined (Jev: 64k);
* ``max_state_question_tokens`` — state plus the single LONGEST question
  (Jev: 32k).

Oversize requests must fail fast with Jev-shaped ``422``, the same way
unknown models and malformed questions do. Current gaps this fixture pins:

1. Oversize state/questions are silently accepted (``200``). Downstream,
   Laya's ``build_sequence`` (``laya/common.py``) quietly truncates the
   state to ``max_len`` — accuracy degrades with no signal to the caller.
2. Option sets overflowing the per-question head budget raise
   ``ValueError("question %r options exceed head_max_len=%d")`` from
   ``Agent.system_one`` (``laya/agent.py``), which surfaces as ``500``.
   That is a client-fixable validation problem and must be ``422``.

``BudgetBackend`` below is a weight-free stand-in for ``LayaBackend``:
same overflow signals, configurable budgets so tests stay fast. Token
counting here is naive word splitting (documented approximation);
``LayaBackend.count`` uses a resident checkpoint tokenizer when one is
loaded and the same word approximation otherwise.

Enforcement lives in ``service.enforce_budgets`` (pre-inference ``422``)
plus a ``ValueError`` → ``CompatError`` mapping for head-budget overflow.
"""

import pytest
from fastapi.testclient import TestClient

from laya_serve.app import create_app
from laya_serve.settings import Settings


def _words(text) -> int:
    """Naive token stand-in: whitespace-separated words of the rendered text."""
    import json

    if not isinstance(text, str):
        text = json.dumps(text)
    return len(text.split())


def _question_words(qdef: dict) -> int:
    """Rendered size of one question: instructions + criteria descriptions."""
    total = _words(qdef.get("instructions", ""))
    criteria = qdef.get("criteria")
    if isinstance(criteria, dict):
        total += sum(_words(v) for v in criteria.values() if v is not None)
    elif isinstance(criteria, list):
        total += sum(_words(level) for level in criteria)
    return total


class BudgetBackend:
    """Weight-free backend mirroring Laya's budget signals.

    * ``max_total_tokens``: state + all questions combined.
    * ``max_state_question_tokens``: state + longest single question.
    * ``head_budget``: per-question head budget (``head_max_len``); option
      overflow raises Laya's real ``ValueError`` message.
    """

    def __init__(
        self,
        serving_model: str = "laya-english",
        max_total_tokens: int = 200,
        max_state_question_tokens: int = 120,
        head_budget: int = 60,
    ) -> None:
        self.serving_model = serving_model
        self.max_total_tokens = max_total_tokens
        self.max_state_question_tokens = max_state_question_tokens
        self.head_budget = head_budget

    def count(self, state, questions: dict) -> tuple[int, int]:
        state_n = _words(state)
        per_question = {qid: _question_words(q) for qid, q in questions.items()}
        longest = max(per_question.values(), default=0)
        return state_n + sum(per_question.values()), state_n + longest

    def predict(self, state, questions: dict) -> dict:
        for qid, qdef in questions.items():
            if qdef["type"] == "choice" and _question_words(qdef) > self.head_budget:
                # Same signal as laya.Agent.system_one on head overflow.
                raise ValueError(f"question {qid!r} options exceed head_max_len={self.head_budget}")
        answers = {
            qid: (
                {"type": "noul", "noul": 0.5}
                if qdef["type"] == "noul"
                else (
                    {
                        "type": "choice",
                        "choice": next(iter(qdef["criteria"])),
                        "probabilities": {o: 1.0 for o in qdef["criteria"]},
                        "confidence": 1.0,
                    }
                    if qdef["type"] == "choice"
                    else {
                        "type": "score",
                        "score": 0.0,
                        "legend": {"0": qdef["criteria"][0], "1": qdef["criteria"][1]},
                        "probabilities": {"0": 0.5, "1": 0.5},
                        "confidence": 0.0,
                    }
                )
            )
            for qid, qdef in questions.items()
        }
        total, _ = self.count(state, questions)
        return {"answers": answers, "usage": {"input_tokens": total, "output_tokens": 0}}


@pytest.fixture()
def budget_client():
    backend = BudgetBackend()
    return TestClient(create_app(Settings(), backend), raise_server_exceptions=False), backend


def _words_n(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_within_budget_passes(budget_client):
    """Control: the harness itself works — small requests stay 200."""
    client, _ = budget_client
    response = client.post(
        "/v1/systemone",
        json={
            "state": "short state",
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
        },
    )
    assert response.status_code == 200, response.text


def test_oversize_total_is_422(budget_client):
    """State + all questions over budget must be 422, not silent 200."""
    client, backend = budget_client
    response = client.post(
        "/v1/systemone",
        json={
            "state": _words_n(backend.max_total_tokens + 1),
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
        },
    )
    assert response.status_code == 422, response.text
    assert set(response.json()) == {"error"}


def test_oversize_state_plus_longest_question_is_422(budget_client):
    """State + longest question over budget must be 422, not silent 200."""
    client, backend = budget_client
    half = backend.max_state_question_tokens // 2
    response = client.post(
        "/v1/systemone",
        json={
            "state": _words_n(half + 10),
            "model": "jev-latest",
            "questions": {
                "q1": {"type": "noul", "instructions": _words_n(half + 10)},
                "q2": {"type": "noul", "instructions": "tiny?"},
            },
        },
    )
    assert response.status_code == 422, response.text
    assert set(response.json()) == {"error"}


def test_head_budget_overflow_is_422_not_500(budget_client):
    """255 schema-valid options with fat descriptions must be 422, not 500."""
    from laya_serve.schemas import ErrorResponse

    client, backend = budget_client
    per_option = backend.head_budget // 10 + 1
    criteria = {f"o{i}": _words_n(per_option) for i in range(200)}
    response = client.post(
        "/v1/systemone",
        json={
            "state": "hi",
            "model": "jev-latest",
            "questions": {
                "big": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": criteria,
                }
            },
        },
    )
    assert response.status_code == 422, response.text
    ErrorResponse.model_validate(response.json())


def test_head_value_error_maps_to_422_when_within_token_budgets():
    """Exercise the ValueError path itself: token budgets pass, head overflows."""
    from laya_serve.schemas import ErrorResponse

    backend = BudgetBackend(max_total_tokens=10**9, max_state_question_tokens=10**9, head_budget=5)
    client = TestClient(create_app(Settings(), backend), raise_server_exceptions=False)
    response = client.post(
        "/v1/systemone",
        json={
            "state": "hi",
            "model": "jev-latest",
            "questions": {
                "big": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": {f"o{i}": "some description here" for i in range(10)},
                }
            },
        },
    )
    assert response.status_code == 422, response.text
    payload = response.json()
    ErrorResponse.model_validate(payload)
    assert payload["error"]["field"] == "questions.big.criteria"


def test_non_budget_value_error_stays_500():
    """A ValueError without budget markers is a backend bug, not a 422."""

    class BugBackend:
        serving_model = "laya-english"

        def predict(self, state, questions):
            raise ValueError("unexpected backend bug")

    client = TestClient(create_app(Settings(), BugBackend()), raise_server_exceptions=False)
    response = client.post(
        "/v1/systemone",
        json={
            "state": "hi",
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
        },
    )
    assert response.status_code == 500, response.text
    assert set(response.json()) == {"error"}


def test_backends_without_budget_interface_skip_enforcement():
    """Pre-budget backends keep working: no count/budgets means no 422."""
    from laya_serve.service import enforce_budgets

    class LegacyBackend:
        serving_model = "laya-english"

        def predict(self, state, questions):
            raise AssertionError("must not be called")

    enforce_budgets(
        "hi " * 100000, {"q": {"type": "noul", "instructions": "x?" * 100000}}, LegacyBackend()
    )  # must not raise


def test_count_request_word_counts():
    from laya_serve.inference import count_request

    total, longest = count_request(
        "s1 s2",
        {
            "a": {
                "type": "choice",
                "instructions": "i1",
                "criteria": {"x": "d1 d2", "y": None},
            },
            "b": {"type": "score", "instructions": "i2", "criteria": ["l1", "l2 l3"]},
        },
    )
    assert (total, longest) == (2 + 3 + 4, 2 + 4)


def test_laya_count_falls_back_without_resident_agent():
    from laya_serve.inference import LayaBackend, count_request

    backend = LayaBackend.__new__(LayaBackend)
    backend.serving_model = "laya-english"
    backend._router = None
    state = {"message": "hello world"}
    questions = {"q": {"type": "noul", "instructions": "Urgent?"}}
    assert backend.count(state, questions) == count_request(state, questions)
