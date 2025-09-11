"""Caching functionality for nflreadpy.

This module provides intelligent caching capabilities for NFL data to improve performance
and reduce network requests. It supports both memory and filesystem caching with
configurable expiration and cache modes.

The caching system is designed to be transparent to users - data functions automatically
check the cache before downloading and store results after successful downloads.

Key Features:
- Memory caching for fast repeated access
- Filesystem caching for persistence across sessions
- Configurable cache duration and storage location
- Automatic cache expiration and cleanup
- Pattern-based cache clearing

Examples:
    >>> import nflreadpy as nfl
    >>> 
    >>> # Data is automatically cached
    >>> pbp = nfl.load_pbp([2023])
    >>> 
    >>> # Subsequent calls use cached data
    >>> pbp_again = nfl.load_pbp([2023])  # Much faster!
    >>> 
    >>> # Clear specific cached data
    >>> nfl.clear_cache("pbp_2023")
    >>> 
    >>> # Clear all cached data
    >>> nfl.clear_cache()
"""

import hashlib
import time
from pathlib import Path

import polars as pl

from .config import CacheMode, get_config


class CacheManager:
    """Manages caching for nflreadpy data.
    
    The CacheManager handles both memory and filesystem caching of NFL data to improve 
    performance and reduce network requests. It supports configurable cache modes and 
    automatic expiration of cached data.
    
    Attributes:
        _memory_cache: Internal dictionary storing cached DataFrames with timestamps.
    """

    def __init__(self) -> None:
        """Initialize a new CacheManager instance."""
        self._memory_cache: dict[str, tuple[pl.DataFrame, float]] = {}

    def _get_cache_key(self, url: str, **kwargs: str | int | float | bool) -> str:
        """Generate a unique cache key from URL and parameters.
        
        Args:
            url: The data source URL.
            **kwargs: Additional parameters that affect the data.
            
        Returns:
            MD5 hash string to use as cache key.
        """
        key_string = f"{url}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_file_path(self, cache_key: str) -> Path:
        """Get the filesystem path for storing cached data.
        
        Args:
            cache_key: The unique cache identifier.
            
        Returns:
            Path to the cache file (creates directory if needed).
        """
        config = get_config()
        cache_dir = config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.parquet"

    def get(self, url: str, **kwargs: str | int | float | bool) -> pl.DataFrame | None:
        """Retrieve cached data if available and not expired.
        
        Args:
            url: The data source URL.
            **kwargs: Additional parameters that were used when caching.
            
        Returns:
            Cached DataFrame if available and valid, None otherwise.
            
        Note:
            Checks memory cache first (if using MEMORY mode), then filesystem cache.
            Automatically removes expired cache entries.
        """
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
        """Store data in the cache.
        
        Args:
            url: The data source URL.
            data: The DataFrame to cache.
            **kwargs: Additional parameters that affect the data.
            
        Note:
            Storage location depends on cache mode configuration:
            - MEMORY: Stores in memory with timestamp
            - FILESYSTEM: Saves as Parquet file with current timestamp
            - OFF: No caching performed
        """
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
        """Clear cache entries matching a pattern.
        
        Args:
            pattern: Optional string pattern to match against cache keys.
                   If None, clears all cache entries.
                   
        Examples:
            >>> cache_manager = get_cache_manager()
            >>> cache_manager.clear()  # Clear all cache
            >>> cache_manager.clear("pbp_2023")  # Clear entries containing "pbp_2023"
            
        Note:
            Clears both memory and filesystem cache entries that match the pattern.
        """
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
        """Get cache size and entry count information.
        
        Returns:
            Dictionary containing cache statistics:
            - memory_entries: Number of entries in memory cache
            - filesystem_entries: Number of files in filesystem cache (if enabled)
            - filesystem_size_mb: Total size of filesystem cache in MB (if enabled)
            
        Examples:
            >>> cache_manager = get_cache_manager()
            >>> stats = cache_manager.size()
            >>> print(f"Memory entries: {stats['memory_entries']}")
            >>> print(f"Disk entries: {stats.get('filesystem_entries', 0)}")
        """
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
    """Get the global cache manager instance.
    
    Returns:
        The singleton CacheManager instance used by all nflreadpy functions.
        
    Examples:
        >>> cache_manager = get_cache_manager()
        >>> cache_stats = cache_manager.size()
        >>> cache_manager.clear("pbp_2023")
    """
    return _cache_manager


def clear_cache(pattern: str | None = None) -> None:
    """Clear cached data entries matching a pattern.
    
    This is the main function for clearing nflreadpy's cache. It provides a simple
    interface to the underlying CacheManager functionality.
    
    Args:
        pattern: Optional string pattern to match against cached data.
               If None, clears all cached data. Pattern matching is performed
               on cache keys, which typically contain URLs and parameters.
               
    Examples:
        >>> import nflreadpy as nfl
        >>> nfl.clear_cache()  # Clear all cached data
        >>> nfl.clear_cache("pbp_2023")  # Clear 2023 play-by-play data
        >>> nfl.clear_cache("roster")  # Clear all roster data
        
    Note:
        This affects both memory and filesystem cache depending on your
        cache configuration. See nflreadpy.config for cache settings.

    See Also:
        [nflreadr clear_cache reference](https://nflreadr.nflverse.com/reference/clear_cache.html)
    """
    _cache_manager.clear(pattern)
