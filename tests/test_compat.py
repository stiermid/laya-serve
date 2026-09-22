"""Compat golden vectors, drawn from the Jev docs examples.

Covers: choice triage answer, score severity answer, noul answer, the
Laya-only extras being stripped (``action``, noul ``confidence``), and
all :class:`CompatError` paths.
"""

import pytest

from laya_serve import compat
from laya_serve.compat import CompatError


def test_golden_choice_triage():
    raw = {
        "department": {
            "type": "choice",
            "choice": "returns",
            "confidence": 1.0,
            "probabilities": {"shipping": 0.0, "returns": 1.0, "billing": 0.0},
            "action": {"act_probability": 0.2},
        }
    }
    questions = {
        "department": {
            "type": "choice",
            "criteria": {
                "returns": "Exchanges, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        }
    }
    out = compat.shape_response(
        questions, raw, {"input_tokens": 328, "output_tokens": 34}, "laya-english"
    )
    answer = out["answers"]["department"]
    assert answer == {
        "type": "choice",
        "choice": "returns",
        "probabilities": {"shipping": 0.0, "returns": 1.0, "billing": 0.0},
        "confidence": 1.0,
    }
    assert "action" not in answer


def test_golden_score_severity():
    raw = {
        "bug_severity": {
            "type": "score",
            "score": 1.43,
            "confidence": 0.35,
            "legend": {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"},
            "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
            "action": {"act_probability": 0.7},
        }
    }
    questions = {
        "bug_severity": {
            "type": "score",
            "criteria": ["Cosmetic", "Workaround", "Blocking"],
        }
    }
    out = compat.shape_response(questions, raw, {"input_tokens": 1, "output_tokens": 0}, "m")
    answer = out["answers"]["bug_severity"]
    assert answer["score"] == pytest.approx(0 * 0.0 + 1 * 0.57 + 2 * 0.43, abs=1e-2)
    assert list(answer["probabilities"]) == ["0", "1", "2"]  # stringified keys
    assert answer["legend"] == {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"}
    assert "action" not in answer


def test_golden_noul_strips_confidence():
    raw = {
        "is_urgent": {
            "type": "noul",
            "noul": 0.95,
            "confidence": 0.95,  # Laya-only; Jev has no noul confidence
            "action": {"act_probability": 0.1},
        }
    }
    out = compat.shape_response({"is_urgent": {"type": "noul"}}, raw, {}, "m")
    assert out["answers"]["is_urgent"] == {"type": "noul", "noul": 0.95}


def test_structured_legend_passthrough():
    raw = {
        "q": {
            "type": "score",
            "score": 1.0,
            "confidence": 1.0,
            "probabilities": {"0": 0.0, "1": 1.0},
        }
    }
    levels = [{"what": "low", "examples": ["typo"]}, {"what": "high", "examples": ["outage"]}]
    out = compat.shape_response({"q": {"type": "score", "criteria": levels}}, raw, {}, "m")
    assert out["answers"]["q"]["legend"] == {"0": levels[0], "1": levels[1]}


def test_resolve_model_aliases():
    assert compat.resolve_model("jev-latest", "laya-english") == "laya-english"
    assert compat.resolve_model("jev-1.13.0", "laya-english") == "laya-english"
    assert compat.resolve_model("laya-english", "laya-english") == "laya-english"
    assert compat.resolve_model("custom", "laya-english", frozenset({"custom"})) == "laya-english"
    with pytest.raises(CompatError):
        compat.resolve_model("gpt-5", "laya-english")


def test_backend_answer_mismatch_is_422():
    with pytest.raises(CompatError):
        compat.shape_response(
            {"a": {"type": "noul"}}, {"b": {"type": "noul", "noul": 0.1}}, {}, "m"
        )


def test_bad_choice_selection_is_422():
    with pytest.raises(CompatError):
        compat.shape_choice(
            "q",
            {"choice": "zzz", "probabilities": {"a": 1.0}, "confidence": 1.0},
            ["a"],
        )


def test_score_out_of_range_is_422():
    with pytest.raises(CompatError):
        compat.shape_score(
            "q",
            {"score": 5.0, "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.0},
            ["lo", "hi"],
        )


def test_nonfinite_numbers_rejected():
    with pytest.raises(CompatError):
        compat.shape_noul("q", {"noul": float("nan")})
