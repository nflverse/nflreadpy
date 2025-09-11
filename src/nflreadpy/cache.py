"""Caching functionality for nflreadpy."""

import hashlib
import time
from pathlib import Path

import polars as pl

from .config import CacheMode, get_config


class CacheManager:
    """Manages caching for nflreadpy data."""

    def __init__(self) -> None:
        self._memory_cache: dict[str, tuple[pl.DataFrame, float]] = {}

    def _get_cache_key(self, url: str, **kwargs: str | int | float | bool) -> str:
        """Generate a cache key from URL and parameters."""
        key_string = f"{url}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        config = get_config()
        cache_dir = config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.parquet"

    def get(self, url: str, **kwargs: str | int | float | bool) -> pl.DataFrame | None:
        """Retrieve data from cache."""
        config = get_config()

        if config.cache_mode == CacheMode.OFF:
            return None

        cache_key = self._get_cache_key(url, **kwargs)
        current_time = time.time()

        # Try memory cache first
        if config.cache_mode == CacheMode.MEMORY:
            cached_item = self._memory_cache.get(cache_key)
            if cached_item:
                data, timestamp = cached_item
                if current_time - timestamp < config.cache_duration:
                    return data
                else:
                    # Remove expired item
                    del self._memory_cache[cache_key]

        # Try filesystem cache
        elif config.cache_mode == CacheMode.FILESYSTEM:
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                try:
                    # Check if file is expired based on modification time
                    file_mtime = file_path.stat().st_mtime
                    if current_time - file_mtime < config.cache_duration:
                        return pl.read_parquet(file_path)
                    else:
                        # Remove expired cache file
                        file_path.unlink(missing_ok=True)
                except Exception as e:
                    if config.verbose:
                        print(f"Failed to read cache file {file_path}: {e}")
                    # Remove corrupted cache file
                    file_path.unlink(missing_ok=True)

        return None

    def set(
        self, url: str, data: pl.DataFrame, **kwargs: str | int | float | bool
    ) -> None:
        """Store data in cache."""
        config = get_config()

        if config.cache_mode == CacheMode.OFF:
            return

        cache_key = self._get_cache_key(url, **kwargs)

        # Store in memory cache
        if config.cache_mode == CacheMode.MEMORY:
            self._memory_cache[cache_key] = (data, time.time())

        # Store in filesystem cache
        elif config.cache_mode == CacheMode.FILESYSTEM:
            file_path = self._get_file_path(cache_key)
            try:
                data.write_parquet(file_path)
            except Exception as e:
                if config.verbose:
                    print(f"Failed to write cache file {file_path}: {e}")

    def clear(self, pattern: str | None = None) -> None:
        """Clear cache entries matching pattern (or all if pattern is None)."""
        config = get_config()

        # Clear memory cache
        if pattern is None:
            self._memory_cache.clear()
        else:
            keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._memory_cache[key]

        # Clear filesystem cache
        if config.cache_mode == CacheMode.FILESYSTEM:
            cache_dir = config.cache_dir
            if cache_dir.exists():
                if pattern is None:
                    # Remove all cache files
                    for cache_file in cache_dir.glob("*.parquet"):
                        cache_file.unlink()
                else:
                    # Remove matching cache files
                    for cache_file in cache_dir.glob("*.parquet"):
                        if pattern in cache_file.stem:
                            cache_file.unlink()

    def size(self) -> dict[str, int | float]:
        """Get cache size information."""
        config = get_config()
        result: dict[str, int | float] = {"memory_entries": len(self._memory_cache)}

        if config.cache_mode == CacheMode.FILESYSTEM:
            cache_dir = config.cache_dir
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.parquet"))
                result["filesystem_entries"] = len(cache_files)
                result["filesystem_size_mb"] = sum(
                    f.stat().st_size for f in cache_files
                ) / (1024 * 1024)
            else:
                result["filesystem_entries"] = 0
                result["filesystem_size_mb"] = 0.0

        return result


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return _cache_manager


def clear_cache(pattern: str | None = None) -> None:
    """Clear cache entries.

    See Also:
        <https://nflreadr.nflverse.com/reference/clear_cache.html>
    """
    _cache_manager.clear(pattern)
