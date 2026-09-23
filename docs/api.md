# API

## Endpoints

| Method | Path | Notes |
| ------ | ---- | ----- |
| `POST` | `/v1/systemone` | Jev-compatible evaluation endpoint |
| `GET` | `/v1/models` | Serving model + accepted aliases |
| `GET` | `/healthz` | Liveness probe (not part of the Jev API) |

## `POST /v1/systemone`

Request: `{state, model, questions}`.

- `state`: `str | dict | list` — plain text, a record, or a sequence of records.
- `model`: non-empty string, must be a known Jev name, an extra alias,
  or the serving model id. See [Configuration](configuration.md#model-names).
- `questions`: non-empty map of caller-chosen ids to one question each.

### Question types

**`choice`** — 1–255 options as a map (descriptions may be `null`):

```json
{
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages"}
}
```

**`score`** — 2–10 ordered levels as a list:

```json
{
  "type": "score",
  "instructions": "How frustrated is the customer?",
  "criteria": ["Calm", "Frustrated", "Very angry"]
}
```

**`noul`** — yes/no/uncertain-likelihood in `[0, 1]`:

```json
{
  "type": "noul",
  "instructions": "The message conveys urgency?",
  "criteria": {"true": "time-sensitive", "false": "no urgency"}
}
```

`instructions` and criteria descriptions accept `str | dict | list`
(structured prompts); non-string values are rendered by the backend.
`noul` criteria are optional.

### Answers

Response: `{model, answers, usage}` with one typed answer per question id:

- `choice`: `{type, choice, probabilities, confidence}`.
- `score`: `{type, score, legend, probabilities, confidence}` with
  stringified level keys (`{"0": …}`) on the wire, matching Jev HTTP.
- `noul`: `{type, noul}` — no `confidence` field, matching Jev.

`usage` is `{input_tokens, output_tokens}`. `output_tokens` is always `0`
(Laya never generates tokens); `input_tokens` comes from the backend.

## `GET /v1/models`

Returns the serving checkpoint plus every accepted alias:

```json
{"models": [{"name": "laya-english", "description": "...", "release_date": "unknown"}]}
```

## Errors

All errors use `{"error": {"message", "field"}}`:

| Status | Meaning | Retry? |
| ------ | ------- | ------ |
| `401` | Missing/invalid `Authorization: Bearer <key>` | No |
| `422` | Validation: bad question shape, unknown `model`, empty `questions`, context budget exceeded | No — fix the request |
| `529` | Transient overload (GPU OOM, evicted checkpoint, timeout) with `Retry-After: 1` | Yes, with backoff |
| `500` | Unexpected backend failure; internals are logged server-side, never leaked | No |
| `404`/`405` | Unknown route/method, still in the Jev error shape (never FastAPI `{"detail": …}`) | No |

Validation limits: choice ≤ 255 options, score 2–10 levels, `model` required,
non-empty `questions`. Context budgets mirror Jev's 64k (state + all questions)
and 32k (state + longest question); oversize requests are `422`. Sizes are
counted with the checkpoint tokenizer when one is resident, word approximation
otherwise.
