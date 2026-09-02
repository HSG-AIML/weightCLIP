"""Tests for portable token-cache resolution."""

import os
from pathlib import Path

import pytest

from sane.data.datasets.cached_windowed_dataset import resolve_cache_dir


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SANE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_configured_path_is_used():
    assert resolve_cache_dir("/tmp/some/cache") == Path("/tmp/some/cache")


def test_null_falls_back_to_documented_default():
    # the config comments promise ~/.sane/cache/tokens when cache_dir is null
    assert resolve_cache_dir(None) == Path.home() / ".sane" / "cache" / "tokens"


def test_xdg_is_honoured(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")
    assert resolve_cache_dir(None) == Path("/tmp/xdg/sane/tokens")


def test_env_overrides_configured_path(monkeypatch):
    # the case that matters for published checkpoints: the absolute path is
    # embedded in checkpoint.pt, so the override must beat an explicit value
    monkeypatch.setenv("SANE_CACHE_DIR", "/tmp/override")
    assert resolve_cache_dir("/local/cache2/tokens") == Path("/tmp/override")


def test_env_overrides_null(monkeypatch):
    monkeypatch.setenv("SANE_CACHE_DIR", "/tmp/override")
    assert resolve_cache_dir(None) == Path("/tmp/override")


def test_user_expansion():
    assert resolve_cache_dir("~/cache").is_absolute()