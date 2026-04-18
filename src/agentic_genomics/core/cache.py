"""Tiny JSON-on-disk cache.

Every public bio-database we query has rate limits. The agent tends to
re-query the same variants during iteration, tests, and demos. A
filesystem-backed cache makes repeated runs fast and polite to upstream
APIs.

Deliberately kept tiny (no sqlite, no pickle) so it's easy to inspect and
prune during development.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(os.getenv("AG_CACHE_DIR", ".agentic_genomics_cache"))
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # one week


def _key_to_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / namespace / f"{digest}.json"


def get(namespace: str, key: str, ttl: int = _DEFAULT_TTL_SECONDS) -> Any | None:
    """Fetch a cached value or ``None`` if missing/expired."""
    path = _key_to_path(namespace, key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def put(namespace: str, key: str, value: Any) -> None:
    """Store a JSON-serialisable value under ``(namespace, key)``."""
    path = _key_to_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cached_call(namespace: str, key: str, fn, ttl: int = _DEFAULT_TTL_SECONDS) -> Any:
    """Memoise ``fn()`` at the disk level, keyed by ``(namespace, key)``."""
    hit = get(namespace, key, ttl=ttl)
    if hit is not None:
        return hit
    value = fn()
    put(namespace, key, value)
    return value
