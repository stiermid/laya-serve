# laya-serve

Jev-compatible HTTP server for [Laya](https://huggingface.co/convaiinnovations/laya)
System One decision models. Point any Jev client at this server and get typed
`choice` / `score` / `noul` answers from local Laya weights instead of the
TypeSafe API.

## Quickstart

```bash
pip install "laya-serve[inference]"
LAYA_SERVE_BACKEND=laya LAYA_SERVE_PRELOAD=true laya-serve
```

```bash
curl -X POST localhost:8000/v1/systemone \
  -H "Content-Type: application/json" -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "is_urgent": {"type": "noul", "instructions": "The message conveys urgency?"}
    }
  }'
```

Weight-free mode for API development and CI:

```bash
pip install -e ".[test]"
LAYA_SERVE_BACKEND=fake laya-serve  # deterministic uniform answers
```

## Endpoints

| Method | Path | Notes |
| ------ | ---- | ----- |
| `POST` | `/v1/systemone` | Jev-compatible evaluation endpoint |
| `GET` | `/v1/models` | Serving model + accepted aliases |
| `GET` | `/healthz` | Liveness probe (not part of the Jev API) |

Interactive OpenAPI docs are served by the app itself at `/docs` (Swagger UI)
and `/redoc` once the server is running.

## Next steps

- [Installation](installation.md) — Python, Docker, and extras.
- [Quickstart](quickstart.md) — start the server and ask a question.
- [Configuration](configuration.md) — `LAYA_SERVE_*` variables, backends, auth.
- [API](api.md) — request/response shapes and error codes.
- [Jev compatibility](jev-compatibility.md) — verified divergences.
- [Deployment](deployment.md) — Docker and production checklist.
- [Versioning](versioning.md) — package vs docs vs serving-model vs Jev names.
- [API reference](reference.md) — auto-generated module docs.
