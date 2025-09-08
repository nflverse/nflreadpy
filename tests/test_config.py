"""Test configuration functionality."""

import tempfile
from pathlib import Path

import pytest

from nflreadpy.config import (
    CacheMode,
    DataFormat,
    NflreadpyConfig,
    get_config,
    reset_config,
    update_config,
)


def test_config_defaults():
    """Test default configuration values."""
    config = NflreadpyConfig()

    assert config.cache_mode == CacheMode.FILESYSTEM
    assert config.prefer_format == DataFormat.PARQUET
    assert config.verbose is True
    assert config.timeout == 30


def test_config_from_env(monkeypatch):
    """Test configuration from environment variables."""
    # Clear any existing environment variables first
    monkeypatch.delenv("NFLREADPY_CACHE", raising=False)
    monkeypatch.delenv("NFLREADPY_VERBOSE", raising=False)
    monkeypatch.delenv("NFLREADPY_TIMEOUT", raising=False)

    # Set new environment variables
    monkeypatch.setenv("NFLREADPY_CACHE", "memory")
    monkeypatch.setenv("NFLREADPY_VERBOSE", "false")
    monkeypatch.setenv("NFLREADPY_TIMEOUT", "60")

    # Create a fresh config instance
    config = NflreadpyConfig()

    assert config.cache_mode == CacheMode.MEMORY
    assert config.verbose is False
    assert config.timeout == 60


def test_get_config():
    """Test getting global configuration."""
    config = get_config()
    assert isinstance(config, NflreadpyConfig)


def test_update_config():
    """Test updating configuration."""
    original_verbose = get_config().verbose

    update_config(verbose=not original_verbose)
    assert get_config().verbose == (not original_verbose)

    # Reset for other tests
    reset_config()


def test_update_config_invalid_key():
    """Test updating configuration with invalid key."""
    with pytest.raises(ValueError, match="Unknown configuration option"):
        update_config(invalid_key="value")


def test_reset_config():
    """Test resetting configuration to defaults."""
    # Change a value
    update_config(verbose=False)
    assert get_config().verbose is False

    # Reset
    reset_config()
    assert get_config().verbose is True


def test_cache_dir_creation():
    """Test cache directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = NflreadpyConfig(cache_dir=Path(temp_dir) / "test_cache")

        # Directory should be created when accessed
        assert isinstance(config.cache_dir, Path)
