#!/usr/bin/env bash
# Smoke-test a running laya-serve instance end to end over HTTP.
#
# Usage:
#   scripts/smoke.sh [BASE_URL]      # default: http://localhost:8000
#
# Boots nothing itself: start the server first, e.g.
#   LAYA_SERVE_BACKEND=fake laya-serve &
# Asserts health, models, a Jev-shaped systemone response, and the 422 path.
set -euo pipefail

BASE="${1:-http://localhost:8000}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "== GET /healthz"
[ "$(curl -sf "$BASE/healthz")" = '{"status":"ok"}' ] || fail "healthz"

echo "== GET /v1/models"
curl -sf "$BASE/v1/models" | grep -q 'laya-english' || fail "models listing"

echo "== POST /v1/systemone (happy path)"
RESP="$(curl -sf -X POST "$BASE/v1/systemone" -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team?",
      "criteria": {"billing": "Payments", "technical": "Bugs"}},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"}
  }}')" || fail "systemone request"
echo "$RESP" | python3 -c "
import json, sys
p = json.load(sys.stdin)
assert p['model'], 'missing model'
assert set(p['answers']) == {'department', 'is_urgent'}, p['answers'].keys()
assert set(p['answers']['is_urgent']) == {'type', 'noul'}, 'noul shape'
assert set(p['answers']['department']) == {'type', 'choice', 'probabilities', 'confidence'}, 'choice shape'
assert abs(sum(p['answers']['department']['probabilities'].values()) - 1.0) < 1e-6, 'probs sum'
print('response shape OK')
" || fail "response shape"

echo "== POST /v1/systemone (unknown model -> 422)"
CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/systemone" \
  -H 'Content-Type: application/json' -d '{
    "state": "x", "model": "gpt-5",
    "questions": {"q": {"type": "noul", "instructions": "y?"}}}')"
[ "$CODE" = 422 ] || fail "expected 422, got $CODE"

echo "SMOKE OK"
