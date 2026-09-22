"""HTTP-level tests against the FastAPI app (fake backend, no weights)."""

import pytest
from fastapi.testclient import TestClient

from laya_serve.app import create_app
from laya_serve.inference import FakeBackend
from laya_serve.settings import Settings

TRIAGE_BODY = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency?"},
    },
}


@pytest.fixture()
def client():
    app = create_app(Settings(), FakeBackend("laya-english"))
    return TestClient(app, raise_server_exceptions=False)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_models_lists_serving_model_and_aliases(client):
    names = [m["name"] for m in client.get("/v1/models").json()["models"]]
    assert "laya-english" in names
    assert "jev-latest" in names


def test_systemone_happy_path_is_jev_shaped(client):
    body = client.post("/v1/systemone", json=TRIAGE_BODY)
    assert body.status_code == 200, body.text
    payload = body.json()
    assert payload["model"] == "laya-english"
    assert set(payload["answers"]) == {"department", "frustration", "is_urgent"}

    choice = payload["answers"]["department"]
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}
    assert choice["choice"] in ("billing", "technical", "sales")
    assert abs(sum(choice["probabilities"].values()) - 1.0) < 1e-9

    score = payload["answers"]["frustration"]
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence"}
    assert list(score["probabilities"]) == ["0", "1", "2"]
    assert score["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}

    noul = payload["answers"]["is_urgent"]
    assert set(noul) == {"type", "noul"}  # no confidence, no action
    assert set(payload["usage"]) == {"input_tokens", "output_tokens"}


def test_unknown_model_is_422(client):
    body = dict(TRIAGE_BODY, model="gpt-5")
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "model"


def test_invalid_question_is_422_with_jev_error_shape(client):
    body = dict(TRIAGE_BODY)
    body["questions"] = {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert set(response.json()) == {"error"}


def test_auth_enforced_when_key_configured():
    app = create_app(Settings(api_key="secret"), FakeBackend("laya-english"))
    authed = TestClient(app, raise_server_exceptions=False)
    assert authed.post("/v1/systemone", json=TRIAGE_BODY).status_code == 401
    assert (
        authed.post(
            "/v1/systemone", json=TRIAGE_BODY, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert (
        authed.post(
            "/v1/systemone", json=TRIAGE_BODY, headers={"Authorization": "Bearer secret"}
        ).status_code
        == 200
    )
