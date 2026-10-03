"""nflverse metadata (nflverse_type, nflverse_timestamp) of parquet files."""

import io
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import polars as pl

from .config import get_config

# Skip other footer keys such as "ARROW:schema" and the R attributes ("r")
METADATA_PREFIX = "nflverse_"

# nflverse timestamps are in America/Toronto time
_UTC_OFFSET_HOURS = {"EST": -5, "EDT": -4}

# Metadata per DataFrame, keyed by id() with a weak reference to the frame
_registry: dict[int, tuple[weakref.ref[pl.DataFrame], dict[str, str]]] = {}

# Set by quiet_metadata()
_quiet: ContextVar[bool] = ContextVar("nflreadpy_quiet_metadata", default=False)


def read_file_metadata(source: bytes | str | Path) -> dict[str, str]:
    """Read the nflverse_* key-value metadata of a parquet file.

    Args:
        source: Parquet file content or path.

    Returns:
        The nflverse_* entries, or an empty dict if there are none or Polars
        is older than 1.30.
    """
    reader = getattr(pl, "read_parquet_metadata", None)
    if reader is None:
        return {}

    try:
        file = io.BytesIO(source) if isinstance(source, bytes) else source
        raw = reader(file)
    except Exception:
        # Metadata is optional
        return {}

    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).startswith(METADATA_PREFIX)
    }


def attach_metadata(df: pl.DataFrame, metadata: dict[str, str]) -> pl.DataFrame:
    """Store metadata for a DataFrame. An empty dict removes it."""
    key = id(df)
    if not metadata:
        _registry.pop(key, None)
        return df

    def _forget(ref: weakref.ref[pl.DataFrame]) -> None:
        entry = _registry.get(key)
        if entry is not None and entry[0] is ref:
            del _registry[key]

    try:
        ref = weakref.ref(df, _forget)
    except TypeError:
        # Frame cannot be weakly referenced
        return df

    _registry[key] = (ref, dict(metadata))
    return df


def copy_metadata(source: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """Copy the metadata of `source` to `target` and return `target`.

    Loaders that combine several files pass the last one, like
    `rbindlist_with_attrs()` in nflreadr.
    """
    return attach_metadata(target, get_metadata(source))


def get_metadata(df: pl.DataFrame) -> dict[str, str]:
    """
    Get the nflverse metadata of a DataFrame returned by a load function.

    Args:
        df: A DataFrame returned by one of the `load_*()` functions.

    Returns:
        Dictionary with `nflverse_type` and `nflverse_timestamp`. Empty for
        csv files, older parquet files and frames not loaded by nflreadpy.

    Examples:
        >>> import nflreadpy as nfl
        >>> pbp = nfl.load_pbp(2025)
        >>> nfl.get_metadata(pbp)
        {'nflverse_timestamp': '2026-08-13 08:26:02 EDT',
         'nflverse_type': 'play by play data'}

    Note:
        With several seasons the metadata of the last file is returned, like
        nflreadr. Frames you create with `filter()`, `pl.concat()` etc. have
        no metadata.
    """
    entry = _registry.get(id(df))
    if entry is None:
        return {}

    ref, metadata = entry
    if ref() is not df:
        return {}

    return dict(metadata)


def format_timestamp(timestamp: str, tz: tzinfo | None = None) -> str:
    """Convert an nflverse timestamp to `tz` (default: local timezone).

    Args:
        timestamp: Timestamp as stored, e.g. "2026-08-13 08:26:02 EDT".
        tz: Timezone to convert to.

    Returns:
        Formatted timestamp, e.g. "2026-08-13 14:26:02 CEST", or the input
        unchanged if it cannot be converted.
    """
    parts = timestamp.strip().rsplit(" ", 1)
    if len(parts) != 2 or parts[1] not in _UTC_OFFSET_HOURS:
        return timestamp

    try:
        parsed = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp

    offset = timezone(timedelta(hours=_UTC_OFFSET_HOURS[parts[1]]))
    try:
        converted = parsed.replace(tzinfo=offset).astimezone(tz)
        name = converted.tzname() or ""
        if not name or not name.isascii() or " " in name:
            # e.g. "W. Europe Daylight Time" on Windows
            name = converted.strftime("UTC%z")
        return f"{converted:%Y-%m-%d %H:%M:%S} {name}"
    except (OverflowError, OSError, ValueError):
        # Out of range
        return timestamp


def metadata_message(metadata: dict[str, str]) -> str | None:
    """Build the "Data updated" line, or None if there is no timestamp."""
    timestamp = metadata.get("nflverse_timestamp")
    if not timestamp:
        return None

    message = f"Data updated: {format_timestamp(timestamp)}"

    data_type = metadata.get("nflverse_type")
    if not data_type:
        return message
    if not data_type.lower().startswith("nflverse"):
        data_type = f"nflverse {data_type}"
    return f"{data_type} | {message}"


@contextmanager
def quiet_metadata() -> Iterator[None]:
    """Suppress the "Data updated" line for internal downloads."""
    token = _quiet.set(True)
    try:
        yield
    finally:
        _quiet.reset(token)


def announce_metadata(df: pl.DataFrame) -> None:
    """Print the "Data updated" line if show_metadata is on.

    If show_metadata is not set, it follows verbose.
    """
    config = get_config()
    show = config.verbose if config.show_metadata is None else config.show_metadata
    if not show or _quiet.get():
        return

    message = metadata_message(get_metadata(df))
    if message is not None:
        print(message)
