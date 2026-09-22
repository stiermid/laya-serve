FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LAYA_SERVE_BACKEND=laya \
    LAYA_SERVE_PRELOAD=true

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[inference]"

EXPOSE 8000
CMD ["laya-serve"]
