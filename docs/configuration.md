# Configuration

All settings use the `LAYA_SERVE_` environment prefix
(see `laya_serve.settings.Settings`).

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `LAYA_SERVE_BACKEND` | `fake` | `laya` (real weights) or `fake` (weight-free, deterministic) |
| `LAYA_SERVE_SERVING_MODEL` | `laya-english` | Id reported in the `model` response field |
| `LAYA_SERVE_EXTRA_MODELS` | `` | Extra accepted `model` names, comma-separated |
| `LAYA_SERVE_DEVICE` | auto | Passed to the Laya `Router` (`cuda`, `cpu`, …) |
| `LAYA_SERVE_MAX_LOADED` | `1` | Router LRU cap on resident checkpoints |
| `LAYA_SERVE_PRELOAD` | `false` | Preload all checkpoints (recommended for servers) |
| `LAYA_SERVE_API_KEY` | unset | When set, requires `Authorization: Bearer <key>` |
| `LAYA_SERVE_RATE_LIMIT_PER_MINUTE` | `0` | `POST /v1/systemone` limit per 60s window per client; `0` disables |

## Backends

- `fake`: uniform `choice`/`score` distributions, `0.5` for `noul`.
  For tests, local API development, and CI. No `torch`/`laya` import.
- `laya`: wraps `laya.Router` (lazy import). Set `DEVICE`, `MAX_LOADED`,
  and `PRELOAD=true` for servers to avoid cold-start latency.

Unknown `LAYA_SERVE_BACKEND` values fail fast at startup with a clear error.

## Model names

Accepted `model` values out of the box:

`jev-latest`, `jev-preview`, `jev-1.13.0`, `jev-1.13`, `laya`, `laya-latest`,
plus the serving model id itself (default `laya-english`).

Anything else is a `422` with `{"error": {"message", "field": "model"}}`,
so client typos fail fast like Jev. Extend the set without code changes:

```bash
LAYA_SERVE_EXTRA_MODELS="jev-1.12,my-alias" laya-serve
```

The response `model` field always echoes the *serving* checkpoint
(e.g. `laya-english`), mirroring how Jev echoes the resolved version id.

## Auth

Unset `LAYA_SERVE_API_KEY` means no auth (local dev default).
When set, every `POST /v1/systemone` requires:

```bash
curl -H "Authorization: Bearer $LAYA_SERVE_API_KEY" ...
```

Missing or wrong keys return `401` in the Jev error shape.
`/healthz` and `/v1/models` stay unauthenticated.

## Rate limiting

`LAYA_SERVE_RATE_LIMIT_PER_MINUTE=60` allows 60 `POST /v1/systemone`
requests per 60s window per client identity (the `Authorization` value when
`API_KEY` is set, else client IP). Excess requests return Jev-shaped `429`
with `Retry-After`. `0` (default) disables in-process limiting. Counters are
per-process; multi-worker deployments should enforce limits at the gateway.
