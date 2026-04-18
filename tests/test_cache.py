"""Tests for the on-disk cache."""

from __future__ import annotations

import os
import time

import pytest

from agentic_genomics.core import cache


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AG_CACHE_DIR", str(tmp_path))
    # Module-level _CACHE_DIR was captured at import time; patch it directly.
    monkeypatch.setattr(cache, "_CACHE_DIR", tmp_path)
    yield


def test_put_then_get_round_trip():
    cache.put("ns", "key", {"hello": "world"})
    assert cache.get("ns", "key") == {"hello": "world"}


def test_miss_returns_none():
    assert cache.get("ns", "missing") is None


def test_cached_call_is_memoised():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"v": calls["n"]}

    first = cache.cached_call("ns", "k", fn)
    second = cache.cached_call("ns", "k", fn)
    assert first == second
    assert calls["n"] == 1


def test_ttl_expiry():
    cache.put("ns", "k", {"v": 1})
    path = cache._key_to_path("ns", "k")
    # Backdate the file so the TTL check treats it as expired.
    old = time.time() - 10_000
    os.utime(path, (old, old))
    assert cache.get("ns", "k", ttl=1) is None
