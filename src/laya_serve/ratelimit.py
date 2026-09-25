"""In-memory fixed-window rate limiter for Jev-compatible ``429`` responses.

Jev returns ``429 Too Many Requests`` when a client exceeds its rate limit;
SDKs retry with exponential backoff. This module provides a minimal,
dependency-free limiter suitable for single-process deployments.

Multi-worker / multi-replica deployments should enforce limits at the
reverse proxy / gateway instead (see ``docs/deployment.md``); per-process
counters do not coordinate across workers.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Fixed-window counter keyed by client identity.

    * ``requests_per_minute <= 0`` means disabled (allow everything).
    * Windows are 60-second fixed windows anchored on first use per key.
    * Thread-safe via a single lock; counters are tiny dicts.
    """

    def __init__(self, requests_per_minute: int = 0) -> None:
        self.requests_per_minute = max(0, int(requests_per_minute))
        self._lock = threading.Lock()
        # key -> (window_start_monotonic, count_in_window)
        self._windows: dict[str, tuple[float, int]] = {}

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record one hit for ``key``.

        Returns ``(allowed, retry_after_seconds)``. When ``allowed`` is
        ``False``, ``retry_after_seconds`` is the whole seconds until the
        current window resets (>= 1).
        """
        if not self.enabled:
            return True, 0
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            start, count = self._windows.get(key, (timestamp, 0))
            if timestamp - start >= 60.0:
                start, count = timestamp, 0
            count += 1
            self._windows[key] = (start, count)
            if count <= self.requests_per_minute:
                return True, 0
            retry_after = max(1, int(60.0 - (timestamp - start)) + 1)
            return False, retry_after


class RateLimitExceeded(Exception):
    """Raised when a request exceeds the rate limit; handled as Jev ``429``."""

    def __init__(self, retry_after: int = 60) -> None:
        super().__init__("Rate limit exceeded. Back off and retry after a short delay.")
        self.retry_after = max(1, int(retry_after))
