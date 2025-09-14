"""Configuration management for nflreadpy."""

from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheMode(str, Enum):
    """Cache modes for data storage.

    Attributes:
        MEMORY: Cache data in memory (faster, but cleared on restart)
        FILESYSTEM: Cache data to disk (persistent across restarts)
        OFF: Disable caching entirely
    """

    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    OFF = "off"


class DataFormat(str, Enum):
    """Preferred data format for downloads.

    Attributes:
        PARQUET: Apache Parquet format (recommended - faster and more efficient)
        CSV: Comma-separated values format (universal compatibility)
    """

    PARQUET = "parquet"
    CSV = "csv"


class NflreadpyConfig(BaseSettings):
    """Configuration settings for nflreadpy.

    This class manages all configuration options for the nflreadpy package.
    Settings can be configured via environment variables or programmatically.

    Environment Variables:
        - NFLREADPY_CACHE: Cache mode ("memory", "filesystem", or "off")
        - NFLREADPY_CACHE_DIR: Directory path for filesystem cache
        - NFLREADPY_CACHE_DURATION: Cache duration in seconds
        - NFLREADPY_PREFER: Preferred data format ("parquet" or "csv")
        - NFLREADPY_VERBOSE: Enable verbose output (true/false)
        - NFLREADPY_TIMEOUT: HTTP request timeout in seconds
        - NFLREADPY_USER_AGENT: Custom user agent string

    Example:
        ```python
        from nflreadpy.config import update_config, get_config

        # Update settings programmatically
        update_config(cache_mode="filesystem", verbose=False)

        # Get current settings
        config = get_config()
        print(f"Cache mode: {config.cache_mode}")
        ```
    """

    # Cache settings
    cache_mode: CacheMode = Field(
        default=CacheMode.MEMORY,
        description="Cache mode for storing downloaded data. 'memory' caches in RAM (fast but temporary), 'filesystem' saves to disk (persistent), 'off' disables caching.",
        alias="NFLREADPY_CACHE",
    )

    cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("nflreadpy")),
        description="Directory path for filesystem cache storage. Only used when cache_mode is 'filesystem'. Defaults to system cache directory.",
        alias="NFLREADPY_CACHE_DIR",
    )

    cache_duration: int = Field(
        default=86400,
        description="How long to keep cached data before re-downloading, in seconds. Default is 86400 (24 hours). Set to 0 to always refresh.",
        alias="NFLREADPY_CACHE_DURATION",
    )

    # Progress and logging
    verbose: bool = Field(
        default=True,
        description="Enable verbose output including progress bars and download status messages. Set to False for silent operation.",
        alias="NFLREADPY_VERBOSE",
    )

    # Request settings
    timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds. How long to wait for server responses before giving up. Increase for slow connections.",
        alias="NFLREADPY_TIMEOUT",
    )

    user_agent: str = Field(
        default=f"nflverse/nflreadpy {version('nflreadpy')}",
        description="User agent string sent with HTTP requests. Identifies the client to servers. Default includes package name and version.",
        alias="NFLREADPY_USER_AGENT",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# Global configuration instance
config = NflreadpyConfig()


def get_config() -> NflreadpyConfig:
    """Get the current configuration instance.

    Returns:
        The global configuration object containing all current settings.

    Example:
        ```python
        config = get_config()
        print(f"Cache directory: {config.cache_dir}")
        print(f"Verbose mode: {config.verbose}")
        ```
    """
    return config


def update_config(**kwargs: Any) -> None:
    """Update configuration settings programmatically.

    Args:
        **kwargs: Configuration options to update. Valid options include:
            - cache_mode: "memory", "filesystem", or "off"
            - cache_dir: Path to cache directory (str or Path)
            - cache_duration: Cache duration in seconds (int)
            - verbose: Enable verbose output (bool)
            - timeout: HTTP timeout in seconds (int)
            - user_agent: Custom user agent string (str)

    Raises:
        ValueError: If an unknown configuration option is provided.

    Example:
        ```python
        # Enable filesystem caching with custom directory
        update_config(
            cache_mode="filesystem",
            cache_dir="/path/to/my/cache",
            verbose=True
        )

        # Disable caching and increase timeout
        update_config(
            cache_mode="off"
            timeout=60
        )
        ```
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")


def reset_config() -> None:
    """Reset all configuration settings to their default values.

    This will restore all settings to their initial state, clearing any
    programmatic or environment variable overrides.

    Example:
        ```python
        # Make some changes
        update_config(cache_mode="off", verbose=False)

        # Reset everything back to defaults
        reset_config()

        # Now cache_mode is "memory" and verbose is True again
        ```
    """
    global config
    config = NflreadpyConfig()
