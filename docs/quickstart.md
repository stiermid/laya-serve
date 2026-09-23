# Quickstart

## Start the server

Real weights:

```bash
LAYA_SERVE_BACKEND=laya LAYA_SERVE_PRELOAD=true laya-serve
# listening on http://0.0.0.0:8000
```

Weight-free (deterministic uniform answers, no model download):

```bash
LAYA_SERVE_BACKEND=fake laya-serve
```

## Ask a question

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
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      },
      "is_urgent": {"type": "noul", "instructions": "The message conveys urgency?"}
    }
  }'
```

Response (shapes vary by backend; the `fake` backend returns uniform probabilities):

```json
{
  "model": "laya-english",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.33, "technical": 0.33, "sales": 0.33},
      "confidence": 0.0
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "confidence": 0.0,
      "probabilities": {"0": 0.33, "1": 0.33, "2": 0.33}
    },
    "is_urgent": {"type": "noul", "noul": 0.5}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

## Check the server

```bash
curl localhost:8000/healthz
# {"status": "ok"}

curl localhost:8000/v1/models
```

Interactive OpenAPI docs (Swagger UI / ReDoc) are available at
`/docs` and `/redoc` while the server is running.

## Next step

Point any Jev client at `http://localhost:8000` instead of the TypeSafe API —
see [API](api.md) for request shapes and [Configuration](configuration.md)
for auth and model aliases.
