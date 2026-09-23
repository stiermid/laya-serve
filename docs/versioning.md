# Versioning

`version` means four different things here. They are independent —
upgrading one does not upgrade the others.

| Axis | Canonical source | Example | Where you see it |
| ---- | ---------------- | ------- | ---------------- |
| Package / release | `pyproject.toml` `version` + git tag `v*` | `0.1.1` | PyPI, `laya_serve.__version__`, `laya-serve --version`, OpenAPI `version` |
| Docs | `mike` versions on `gh-pages` | `dev`, `0.1.1`, `latest` | Version picker at <https://stiermid.github.io/laya-serve/> |
| Serving model | `LAYA_SERVE_SERVING_MODEL` | `laya-english` | `model` response field, `GET /v1/models` |
| Jev compat | `laya_serve.compat.JEV_KNOWN_MODELS` + `LAYA_SERVE_EXTRA_MODELS` | `jev-1.13.0`, `jev-latest` | Accepted `model` request values |

!!! warning "Three different `latest`s"
    - `latest` (docs): `mike` alias to the newest `X.Y.Z` docs build.
    - `jev-latest` (request): accepted Jev model name, resolves to the serving checkpoint.
    - `laya-latest` (request): same, Laya-side alias. See [Configuration](configuration.md#model-names).

## Package / release

`pyproject.toml` is the single source of truth. `laya_serve.__version__`
reads it via `importlib.metadata` (fallback `0.0.0+unknown` for a bare
source checkout), and `create_app()` reuses it for the OpenAPI `version`
so the three can never drift:

```bash
laya-serve --version
# laya-serve 0.1.1
```

Release flow (maintainers):

1. Bump `version` in `pyproject.toml`, commit as `chore(release): bump version to X.Y.Z`.
2. Tag `vX.Y.Z` and push the tag. CI publishes that exact tag to PyPI
   (`.github/workflows/ci.yml`) and deploys docs for `X.Y.Z` (below).
3. Never hand-edit `__version__` or the FastAPI `version` — they follow `pyproject.toml`.

`tests/test_version.py` guards `__version__ == importlib.metadata` and
`app.version == __version__`.

## Docs (`mike`)

CI (`.github/workflows/docs.yml`) owns `gh-pages`; do not push it by hand:

- `master` push → `mike deploy dev` (default is `dev` until the first
  stable tag exists, then `latest` takes over and stays default).
- Tag `vX.Y.Z` push → `mike deploy --update-aliases X.Y.Z latest`
  (the leading `v` is stripped) and `latest` becomes the default.

Local preview is unversioned and never pushes:

```bash
uv sync --extra docs
uv run zensical serve        # live preview of this checkout
uv run zensical build --strict  # same check CI runs
```

Versioned preview only when you need to check aliases:

```bash
uv run mike deploy dev        # no --push: local gh-pages branch only
uv run mike serve             # browse dev / latest / X.Y.Z picker locally
uv run mike delete dev        # clean up when done
```

## Serving model vs Jev model names

- The response `model` field always echoes the *serving* checkpoint
  (`LAYA_SERVE_SERVING_MODEL`, default `laya-english`), mirroring how Jev
  echoes the resolved version id.
- The request `model` field accepts the serving id plus the Jev family
  (`jev-latest`, `jev-preview`, `jev-1.13.0`, `jev-1.13`, `laya`,
  `laya-latest`) plus `LAYA_SERVE_EXTRA_MODELS`. Unknown names are `422`.
- Changing `LAYA_SERVE_SERVING_MODEL` changes what clients see in
  responses and `GET /v1/models`; it does not change the package or docs
  version.
