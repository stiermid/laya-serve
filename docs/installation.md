# Installation

## Requirements

- Python 3.10+ (3.10, 3.11, 3.12, and 3.13 are tested in CI).
- `pip` or [`uv`](https://docs.astral.sh/uv/).

## From PyPI

Weight-free mode (API development, CI, no GPU/torch needed):

```bash
pip install "laya-serve[test]"
```

Production inference with real Laya weights:

```bash
pip install "laya-serve[inference]"
```

## From source

```bash
git clone https://github.com/stiermid/laya-serve
cd laya-serve
pip install -e ".[test]"      # weight-free
# or
pip install -e ".[inference]" # real weights
```

Reproducible install with `uv` (uses `uv.lock`):

```bash
uv sync --extra test
uv run pytest -q
```

## Extras

| Extra | Pulls in | Use for |
| ----- | -------- | ------- |
| `test` | `pytest`, `httpx` | local dev, CI, `fake` backend |
| `inference` | `laya>=0.3.0`, `torch>=2.0.0`, `transformers>=4.48.0` | `laya` backend with real checkpoints |
| `docs` | `zensical`, `mkdocstrings-python`, `mike` | building this site |

## Docker

```bash
docker build -t laya-serve .
docker run -p 8000:8000 laya-serve
```

The image defaults to `LAYA_SERVE_BACKEND=laya` and `LAYA_SERVE_PRELOAD=true`.
See [Deployment](deployment.md) for production settings.
