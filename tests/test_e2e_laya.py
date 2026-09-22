"""End-to-end test against real Laya weights (opt-in, slow).

Runs only when ALL of the following hold:

* the ``laya`` package (``pip install "laya-serve[inference]"``) is importable,
* ``LAYA_SERVE_RUN_E2E=1`` is set (first run downloads ~800MB of weights),
* ``torch`` can load the checkpoint in this environment.

Otherwise the test skips cleanly, so plain ``pytest`` stays fast and offline.
"""

import os

import pytest

laya = pytest.importorskip("laya", reason="laya weights stack not installed")

pytestmark = pytest.mark.skipif(
    os.environ.get("LAYA_SERVE_RUN_E2E") != "1",
    reason="set LAYA_SERVE_RUN_E2E=1 to run the real-weights e2e test",
)

from fastapi.testclient import TestClient  # noqa: E402

from laya_serve.app import create_app  # noqa: E402
from laya_serve.inference import LayaBackend  # noqa: E402
from laya_serve.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def client():
    settings = Settings(backend="laya", serving_model="laya-english")
    app = create_app(settings, LayaBackend(settings))
    return TestClient(app, raise_server_exceptions=False)


def test_e2e_billing_triage(client):
    """The Jev-docs-style billing ticket must come back Jev-shaped and sane."""
    response = client.post(
        "/v1/systemone",
        json={
            "state": {
                "from": "user@acme.com",
                "subject": "Duplicate charge on invoice #4411",
                "body": "Hi, we were billed twice for March. Please refund the duplicate.",
            },
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this request?",
                    "criteria": {
                        "billing": "invoices, payments, refunds",
                        "technical": "bugs, outages, system errors",
                        "sales": "pricing, new contracts",
                    },
                },
                "refund_requested": {
                    "type": "noul",
                    "instructions": "Does the user explicitly request a refund?",
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this request?",
                    "criteria": ["not urgent", "soon", "critical deadline"],
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    choice = payload["answers"]["department"]
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}
    assert choice["choice"] in ("billing", "technical", "sales")
    assert abs(sum(choice["probabilities"].values()) - 1.0) < 1e-3

    noul = payload["answers"]["refund_requested"]
    assert set(noul) == {"type", "noul"}
    assert 0.0 <= noul["noul"] <= 1.0

    score = payload["answers"]["urgency"]
    assert 0.0 <= score["score"] <= 2.0
    assert list(score["probabilities"]) == ["0", "1", "2"]
