# Jev compatibility

Verified against `docs.typesafe.ai` and `laya` 0.3.x source.
Deliberate, documented divergences:

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
   budget; request sizes use a deterministic word count on every backend
   so the `200`/`422` boundary is stable across cold starts and evictions.
   The checkpoint still truncates per-question sequences at its own `max_len`.
6. **Score `legend`/`probabilities` keys are strings** on the wire
   (`{"0": …}`), matching Jev HTTP.
7. **`choice` criteria as a list** is accepted leniently (mapped to
   `{label: None}`); Jev requires a map.

## Practical notes

- Unknown `model` names are `422` (fail fast on typos, like Jev).
- `429` is enforced in-process on `POST /v1/systemone` when
  `LAYA_SERVE_RATE_LIMIT_PER_MINUTE > 0` (per API key, else per client IP),
  with `Retry-After`; transient pressure surfaces as `529` with
  `Retry-After: 1`. Multi-worker deployments should enforce limits at the
  gateway.
- `500` responses never leak backend internals; details go to server logs.
