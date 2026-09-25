"""Rate-limit (Jev 429) tests: in-process fixed window on POST /v1/systemone."""

from fastapi.testclient import TestClient

from laya_serve.app import create_app
from laya_serve.inference import FakeBackend
from laya_serve.ratelimit import RateLimiter
from laya_serve.settings import Settings

TRIAGE_BODY = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
}


def _client(**settings_kwargs):
    settings = Settings(**settings_kwargs)
    app = create_app(settings, FakeBackend("laya-english"))
    return TestClient(app, raise_server_exceptions=False)


def test_ratelimit_disabled_by_default():
    client = _client()
    for _ in range(5):
        response = client.post("/v1/systemone", json=TRIAGE_BODY)
        assert response.status_code == 200, response.text


def test_ratelimit_allows_burst_then_429_with_retry_after():
    client = _client(rate_limit_per_minute=2)
    assert client.post("/v1/systemone", json=TRIAGE_BODY).status_code == 200
    assert client.post("/v1/systemone", json=TRIAGE_BODY).status_code == 200
    limited = client.post("/v1/systemone", json=TRIAGE_BODY)
    assert limited.status_code == 429, limited.text
    assert limited.headers.get("retry-after") is not None
    assert int(limited.headers["retry-after"]) >= 1
    payload = limited.json()
    assert set(payload) == {"error"}
    assert payload["error"]["field"] is None


def test_ratelimit_error_matches_schema():
    from laya_serve.schemas import ErrorResponse

    client = _client(rate_limit_per_minute=1)
    assert client.post("/v1/systemone", json=TRIAGE_BODY).status_code == 200
    response = client.post("/v1/systemone", json=TRIAGE_BODY)
    assert response.status_code == 429
    ErrorResponse.model_validate(response.json())


def test_ratelimit_does_not_apply_to_healthz_or_models():
    client = _client(rate_limit_per_minute=1)
    assert client.post("/v1/systemone", json=TRIAGE_BODY).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/models").status_code == 200


def test_limiter_unit_fixed_window_and_reset():
    limiter = RateLimiter(requests_per_minute=1)
    assert limiter.check("k", now=0.0) == (True, 0)
    allowed, retry_after = limiter.check("k", now=1.0)
    assert allowed is False
    assert retry_after >= 1
    assert limiter.check("other", now=1.0) == (True, 0)
    assert limiter.check("k", now=61.0) == (True, 0)


def test_limiter_disabled_allows_everything():
    limiter = RateLimiter(requests_per_minute=0)
    assert limiter.enabled is False
    for _ in range(10):
        assert limiter.check("k")[0] is True
