# Development

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

Or with [`uv`](https://docs.astral.sh/uv/) (reproducible, via `uv.lock`):

```bash
uv sync --extra test
uv run pytest -q
```

## Lint and hooks

Lint/format via [`ruff`](https://docs.astral.sh/ruff/) and hooks via
[`pre-commit`](https://pre-commit.com/):

```bash
uvx ruff check src tests && uvx ruff format --check src tests
pre-commit install && pre-commit run --all-files
```

## Docs preview

```bash
uv sync --extra docs
uv run mkdocs serve
# -> http://127.0.0.1:8000
```

Strict build (same as CI):

```bash
uv run mkdocs build --strict
```

Versioned preview with `mike`:

```bash
uv run mike deploy --push dev
uv run mike serve
```

## Backend notes

- The HTTP layer depends only on the `Backend` protocol
  (`laya_serve.inference.Backend`), never on `torch`/`laya` directly.
- `FakeBackend` gives deterministic uniform answers for tests and local API
  work; `LayaBackend` wraps `laya.Router` with a lazy import.
- Compat shaping (`laya_serve.compat`) is pure and backend-agnostic, so it
  is unit-testable without weights.
