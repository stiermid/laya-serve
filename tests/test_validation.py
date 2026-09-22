"""Schema-level validation: Jev limits and structured inputs."""

import pytest

from laya_serve.schemas import SystemOneRequest


def _req(questions):
    return {"state": "some state", "model": "jev-latest", "questions": questions}


def test_accepts_all_three_types_with_structured_fields():
    req = SystemOneRequest.model_validate(
        _req(
            {
                "department": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Pay", "technical": None, "sales": ""},
                },
                "frustration": {
                    "type": "score",
                    "instructions": {"question": "How frustrated?", "focus": "tone"},
                    "criteria": [
                        "Calm",
                        {"what": "Frustrated", "examples": ["sigh"]},
                        ["Very", "angry"],
                    ],
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "time-sensitive", "false": "no urgency"},
                },
                "plain": {"type": "noul", "instructions": "Plain?"},
            }
        )
    )
    assert set(req.questions) == {"department", "frustration", "is_urgent", "plain"}


def test_accepts_object_and_array_state():
    for state in (
        "plain text",
        {"message": "hi", "order_id": "A-104"},
        ["Hi", "My card was charged twice."],
    ):
        req = SystemOneRequest.model_validate(
            {"state": state, "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "x?"}}}
        )
        assert req.state == state


def test_rejects_256_choice_options():
    with pytest.raises(Exception):
        SystemOneRequest.model_validate(
            _req(
                {
                    "q": {
                        "type": "choice",
                        "instructions": "x",
                        "criteria": {f"o{i}": "d" for i in range(256)},
                    }
                }
            )
        )


def test_accepts_255_choice_options():
    req = SystemOneRequest.model_validate(
        _req(
            {
                "q": {
                    "type": "choice",
                    "instructions": "x",
                    "criteria": {f"o{i}": "d" for i in range(255)},
                }
            }
        )
    )
    assert len(req.questions["q"].criteria) == 255


@pytest.mark.parametrize("n_levels", [2, 10])
def test_accepts_score_level_bounds(n_levels):
    req = SystemOneRequest.model_validate(
        _req({"q": {"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(n_levels)]}})
    )
    assert len(req.questions["q"].criteria) == n_levels


@pytest.mark.parametrize("n_levels", [1, 11])
def test_rejects_score_levels_outside_2_to_10(n_levels):
    with pytest.raises(Exception):
        SystemOneRequest.model_validate(
            _req({"q": {"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(n_levels)]}})
        )


def test_rejects_empty_questions_and_missing_model():
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {}})
    with pytest.raises(Exception):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul", "instructions": "y?"}}})


def test_rejects_unknown_question_type():
    with pytest.raises(Exception):
        SystemOneRequest.model_validate(
            _req({"q": {"type": "generate", "instructions": "write a poem"}})
        )
