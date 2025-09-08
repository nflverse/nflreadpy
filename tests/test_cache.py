"""Test caching functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from nflreadpy.cache import CacheManager, clear_cache, get_cache_manager
from nflreadpy.config import CacheMode


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pl.DataFrame(
        {"season": [2023, 2023, 2024], "week": [1, 2, 1], "team": ["KC", "BUF", "KC"]}
    )


@pytest.fixture
def cache_manager():
    """Create a fresh cache manager for testing."""
    return CacheManager()


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_memory_cache_get_set(self, cache_manager, sample_dataframe):
        """Test memory cache get and set operations."""
        with patch("nflreadpy.cache.get_config") as mock_config:
            mock_config.return_value.cache_mode = CacheMode.MEMORY

            url = "https://example.com/data.parquet"

            # Should return None for cache miss
            result = cache_manager.get(url, season=2023)
            assert result is None

            # Set data in cache
            cache_manager.set(url, sample_dataframe, season=2023)

            # Should return data for cache hit
            result = cache_manager.get(url, season=2023)
            assert result is not None
            assert result.equals(sample_dataframe)

    def test_filesystem_cache_get_set(self, cache_manager, sample_dataframe):
        """Test filesystem cache get and set operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("nflreadpy.cache.get_config") as mock_config:
                config_mock = MagicMock()
                config_mock.cache_mode = CacheMode.FILESYSTEM
                config_mock.cache_dir = Path(temp_dir)
                config_mock.verbose = False
                mock_config.return_value = config_mock

                url = "https://example.com/data.parquet"

                # Should return None for cache miss
                result = cache_manager.get(url, season=2023)
                assert result is None

                # Set data in cache
                cache_manager.set(url, sample_dataframe, season=2023)

                # Should return data for cache hit
                result = cache_manager.get(url, season=2023)
                assert result is not None
                assert result.equals(sample_dataframe)

    def test_cache_off(self, cache_manager, sample_dataframe):
        """Test cache disabled mode."""
        with patch("nflreadpy.cache.get_config") as mock_config:
            mock_config.return_value.cache_mode = CacheMode.OFF

            url = "https://example.com/data.parquet"

            # Should always return None
            result = cache_manager.get(url, season=2023)
            assert result is None

            # Set should do nothing
            cache_manager.set(url, sample_dataframe, season=2023)
            result = cache_manager.get(url, season=2023)
            assert result is None

    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation."""
        url1 = "https://example.com/data.parquet"
        url2 = "https://example.com/other.parquet"

        key1 = cache_manager._get_cache_key(url1, season=2023)
        key2 = cache_manager._get_cache_key(url1, season=2024)
        key3 = cache_manager._get_cache_key(url2, season=2023)

        # Different URLs or parameters should generate different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Same URL and parameters should generate same key
        key1_again = cache_manager._get_cache_key(url1, season=2023)
        assert key1 == key1_again

    def test_clear_cache_memory(self, cache_manager, sample_dataframe):
        """Test clearing memory cache."""
        with patch("nflreadpy.cache.get_config") as mock_config:
            mock_config.return_value.cache_mode = CacheMode.MEMORY

            # Add some data to cache
            cache_manager.set("url1", sample_dataframe, season=2023)
            cache_manager.set("url2", sample_dataframe, season=2024)

            # Clear all cache
            cache_manager.clear()

            # Should be empty
            result1 = cache_manager.get("url1", season=2023)
            result2 = cache_manager.get("url2", season=2024)
            assert result1 is None
            assert result2 is None

    def test_cache_size_info(self, cache_manager, sample_dataframe):
        """Test cache size information."""
        with patch("nflreadpy.cache.get_config") as mock_config:
            mock_config.return_value.cache_mode = CacheMode.MEMORY

            # Initially empty
            size_info = cache_manager.size()
            assert size_info["memory_entries"] == 0

            # Add some data
            cache_manager.set("url1", sample_dataframe, season=2023)
            cache_manager.set("url2", sample_dataframe, season=2024)

            size_info = cache_manager.size()
            assert size_info["memory_entries"] == 2


def test_global_cache_manager():
    """Test global cache manager functions."""
    manager = get_cache_manager()
    assert isinstance(manager, CacheManager)

    # Test clear_cache function
    clear_cache("test_pattern")  # Should not raise error
