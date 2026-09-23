# laya-serve

[![CI](https://github.com/stiermid/laya-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/stiermid/laya-serve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-serve.svg)](https://pypi.org/project/laya-serve/)
[![License](https://img.shields.io/github/license/stiermid/laya-serve.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![API](https://img.shields.io/badge/API-Jev--compatible-green.svg)](https://docs.typesafe.ai/api)
[![Docs](https://img.shields.io/badge/docs-zensical-blue.svg)](https://stiermid.github.io/laya-serve/)

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

Without weights (API development, CI):

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

Errors use `{"error": {"message", "field"}}` with Jev status codes
(`401` bad key, `422` validation, `529` transient overload with a
`Retry-After` header, `500` unexpected backend failure with internals
logged server-side; `429` rate-limit handling is still future work).

## Configuration (`LAYA_SERVE_` env prefix)

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `LAYA_SERVE_BACKEND` | `fake` | `laya` (real weights) or `fake` (weight-free) |
| `LAYA_SERVE_SERVING_MODEL` | `laya-english` | Id reported in the `model` response field |
| `LAYA_SERVE_EXTRA_MODELS` | `` | Extra accepted `model` names, comma-separated |
| `LAYA_SERVE_DEVICE` | auto | Passed to the Laya Router (`cuda`, `cpu`, …) |
| `LAYA_SERVE_MAX_LOADED` | `1` | Router LRU cap on resident checkpoints |
| `LAYA_SERVE_PRELOAD` | `false` | Preload all checkpoints (recommended for servers) |
| `LAYA_SERVE_API_KEY` | unset | When set, requires `Authorization: Bearer <key>` |

Accepted `model` names out of the box: `jev-latest`, `jev-preview`,
`jev-1.13.0`, `jev-1.13`, `laya`, `laya-latest`, plus the serving model id
itself. Anything else is a `422` (fail fast on typos, like Jev).

## Jev compatibility notes

Verified against `docs.typesafe.ai` and `laya` 0.3.x source. Deliberate,
documented divergences:

1. **Laya-only fields are stripped**: `action` on every answer, and
   `confidence` on `noul` answers. Responses contain exactly the Jev fields.
2. **`model` echoes the serving checkpoint** (e.g. `laya-english`), the same
   way Jev echoes the resolved version id (`jev-1.13.0`).
3. **`output_tokens` is `0`**. Laya never generates tokens; this layer does
   not invent counts. `input_tokens` is the backend's token count.
4. **Confidence formula differs**: Laya uses entropy-based
   `1 - H(p)/log(k)`; Jev documents a peak-based normalization. Same
   probabilities can yield different `confidence` values — calibrate
   thresholds against Laya, not Jev.
5. **Validation**: choice ≤ 255 options, score 2–10 levels, `model` required,
   non-empty `questions` — all `422`. Context budgets mirror Jev's 64k
   (state + all questions) / 32k (state + longest question) accounting and
   surface as `422`, as do option sets overflowing the per-question head
   budget; request sizes are counted with the checkpoint tokenizer when
   one is loaded, word approximation otherwise. The checkpoint still
   truncates per-question sequences at its own `max_len`.
6. **Score `legend`/`probabilities` keys are strings** on the wire
   (`{"0": …}`), matching Jev HTTP.
7. **`choice` criteria as a list** is accepted leniently (mapped to
   `{label: None}`); Jev requires a map.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

Or with [uv](https://docs.astral.sh/uv/) (reproducible, via `uv.lock`):

```bash
uv sync --extra test
uv run pytest -q
```

Lint/format via [ruff](https://docs.astral.sh/ruff/) and hooks via
[pre-commit](https://pre-commit.com/):

```bash
uvx ruff check src tests && uvx ruff format --check src tests
pre-commit install && pre-commit run --all-files
```

## Documentation

Versioned docs (Zensical + `mike`): <https://stiermid.github.io/laya-serve/>
(`latest` tracks the newest `v*` tag, `dev` tracks `master`).

```bash
uv sync --extra docs
uv run zensical serve        # local preview
uv run zensical build --strict  # same check CI runs
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
