# Deployment

## Bare metal / VM

```bash
pip install "laya-serve[inference]"
LAYA_SERVE_BACKEND=laya LAYA_SERVE_PRELOAD=true laya-serve
# -> http://0.0.0.0:8000
```

`PRELOAD=true` is recommended for servers: it loads all checkpoints at
startup instead of on first request. Tune residency with
`LAYA_SERVE_MAX_LOADED=1` and placement with `LAYA_SERVE_DEVICE=cuda`
(or `cpu`).

Host/port are CLI flags (`laya-serve --host 0.0.0.0 --port 8000`).
Put TLS termination and rate limiting in front (reverse proxy / gateway);
`429` handling in-app is future work.

## Docker

```bash
docker build -t laya-serve .
docker run -p 8000:8000 \
  -e LAYA_SERVE_API_KEY="$LAYA_SERVE_API_KEY" \
  laya-serve
```

The shipped `Dockerfile` sets `LAYA_SERVE_BACKEND=laya` and
`LAYA_SERVE_PRELOAD=true` and installs the `inference` extra.

## Health checks

- Liveness: `GET /healthz` → `{"status": "ok"}`. Use it for
  Kubernetes liveness/readiness probes and load-balancer checks.
- `GET /v1/models` verifies the serving model and alias set after a deploy.

## Configuration checklist

- [ ] `LAYA_SERVE_SERVING_MODEL` matches the deployed checkpoint id.
- [ ] `LAYA_SERVE_EXTRA_MODELS` covers any legacy client model strings.
- [ ] `LAYA_SERVE_API_KEY` set; clients send `Authorization: Bearer <key>`.
- [ ] Logs: `500`s log server-side with tracebacks; `529`s log a warning
      and return `Retry-After: 1` so clients back off.
