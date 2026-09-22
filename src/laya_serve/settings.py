"""Runtime configuration, sourced from environment variables.

All variables use the ``LAYA_SERVE_`` prefix, e.g. ``LAYA_SERVE_PRELOAD=true``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAYA_SERVE_", extra="ignore")

    # Model id reported in the ``model`` response field.
    serving_model: str = "laya-english"
    # Additional model names accepted in the request ``model`` field,
    # beyond the built-in Jev family (comma-separated via env).
    extra_models: str = ""

    @property
    def extra_models_set(self) -> frozenset[str]:
        """Parsed :attr:`extra_models` as a set of names."""
        return frozenset(part.strip() for part in self.extra_models.split(",") if part.strip())

    # Inference backend: "laya" (real weights) or "fake" (deterministic,
    # weight-free answers for tests and local dev).
    backend: str = "fake"

    # Passed through to the Laya Router when backend="laya".
    device: str | None = None
    max_loaded: int = 1
    preload: bool = False

    # When set, clients must send ``Authorization: Bearer <api_key>``.
    # Unset means no auth (local dev default).
    api_key: str | None = None
