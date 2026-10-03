"""Tests for nflverse metadata, without network access."""

import gc
import io
from datetime import datetime, timezone

import polars as pl
import pytest

import nflreadpy as nfl
from nflreadpy import metadata as nfl_metadata
from nflreadpy.config import CacheMode, DataFormat, get_config, update_config
from nflreadpy.downloader import get_downloader

NFLVERSE_TYPE = "play by play data"
NFLVERSE_TIMESTAMP = "2026-08-13 08:26:02 EDT"
NFLVERSE_METADATA = {
    "nflverse_type": NFLVERSE_TYPE,
    "nflverse_timestamp": NFLVERSE_TIMESTAMP,
}


def make_parquet(metadata=None):
    """Create a small parquet file in memory, optionally with key-value metadata."""
    if metadata is not None and not hasattr(pl, "read_parquet_metadata"):
        pytest.skip("Polars version has no parquet key-value metadata support")

    df = pl.DataFrame({"season": [2025, 2025, 2025], "play_id": [1, 2, 3]})
    buffer = io.BytesIO()
    if metadata is None:
        df.write_parquet(buffer)
    else:
        df.write_parquet(buffer, metadata=metadata)
    return buffer.getvalue()


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content):
        self.content = content
        self.headers = {}

    def raise_for_status(self):
        return None


class FakeSession:
    """Serves one file for every request and counts the requests."""

    def __init__(self, monkeypatch, content):
        self.content = content
        self.calls = []
        monkeypatch.setattr(get_downloader().session, "get", self.get)

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponse(self.content)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path):
    """Run every test with caching off, quiet output and an empty cache."""
    config = get_config()
    names = ["cache_mode", "cache_dir", "verbose", "show_metadata"]
    saved = {name: getattr(config, name) for name in names}

    update_config(
        cache_mode=CacheMode.OFF,
        cache_dir=tmp_path,
        verbose=False,
        show_metadata=None,
    )
    nfl.clear_cache()
    yield
    # Reset the memory cache, pytest removes tmp_path
    update_config(cache_mode=CacheMode.OFF)
    nfl.clear_cache()
    update_config(**saved)


class TestReadFileMetadata:
    """Test reading key-value metadata from parquet files."""

    def test_keeps_only_nflverse_keys(self):
        """Test that only the nflverse entries are returned."""
        content = make_parquet({**NFLVERSE_METADATA, "other": "value"})
        assert nfl_metadata.read_file_metadata(content) == NFLVERSE_METADATA

    def test_reads_from_path(self, tmp_path):
        """Test reading metadata from a file on disk."""
        file_path = tmp_path / "play_by_play_2025.parquet"
        file_path.write_bytes(make_parquet(NFLVERSE_METADATA))
        assert nfl_metadata.read_file_metadata(file_path) == NFLVERSE_METADATA

    def test_file_without_metadata(self):
        """Test that files saved without metadata return an empty dict."""
        assert nfl_metadata.read_file_metadata(make_parquet()) == {}

    def test_invalid_file(self):
        """Test that unreadable content returns an empty dict."""
        assert nfl_metadata.read_file_metadata(b"not a parquet file") == {}

    def test_polars_without_metadata_reader(self, monkeypatch):
        """Test Polars versions before 1.30, which cannot read the metadata."""
        content = make_parquet(NFLVERSE_METADATA)
        monkeypatch.delattr(pl, "read_parquet_metadata")
        assert nfl_metadata.read_file_metadata(content) == {}

        FakeSession(monkeypatch, content)
        df = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        assert len(df) == 3
        assert nfl.get_metadata(df) == {}


class TestGetMetadata:
    """Test the metadata registry."""

    def test_unknown_frame(self):
        """Test that a DataFrame not loaded by nflreadpy has no metadata."""
        assert nfl.get_metadata(pl.DataFrame({"a": [1]})) == {}

    def test_attach_and_get(self):
        """Test that attached metadata is returned as a copy."""
        df = pl.DataFrame({"a": [1]})
        assert nfl_metadata.attach_metadata(df, NFLVERSE_METADATA) is df

        result = nfl.get_metadata(df)
        assert result == NFLVERSE_METADATA

        result["nflverse_type"] = "changed"
        assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_empty_metadata_removes_entry(self):
        """Test that attaching empty metadata clears earlier metadata."""
        df = pl.DataFrame({"a": [1]})
        nfl_metadata.attach_metadata(df, NFLVERSE_METADATA)
        nfl_metadata.attach_metadata(df, {})
        assert nfl.get_metadata(df) == {}

    def test_object_without_weakref_support(self):
        """Test that metadata is skipped if the frame cannot be referenced."""
        frame = object()
        assert nfl_metadata.attach_metadata(frame, NFLVERSE_METADATA) is frame
        assert nfl.get_metadata(frame) == {}

    def test_new_frames_have_no_metadata(self):
        """Test that frames created by Polars operations have no metadata."""
        df = pl.DataFrame({"a": [1, 2]})
        nfl_metadata.attach_metadata(df, NFLVERSE_METADATA)
        assert nfl.get_metadata(df.filter(pl.col("a") > 1)) == {}

    def test_metadata_is_released_with_the_frame(self):
        """Test that metadata is dropped when the DataFrame is deleted."""
        df = pl.DataFrame({"a": [1]})
        nfl_metadata.attach_metadata(df, NFLVERSE_METADATA)
        key = id(df)
        assert key in nfl_metadata._registry

        del df
        gc.collect()
        assert key not in nfl_metadata._registry


class TestDataUpdatedMessage:
    """Test the formatting of the "Data updated" line."""

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            ("2026-08-13 08:26:02 EDT", "2026-08-13 12:26:02 UTC"),
            ("2026-01-15 08:26:02 EST", "2026-01-15 13:26:02 UTC"),
            ("2026-08-13 23:30:00 EDT", "2026-08-14 03:30:00 UTC"),
        ],
    )
    def test_format_timestamp(self, timestamp, expected):
        """Test converting nflverse timestamps to another timezone."""
        assert nfl_metadata.format_timestamp(timestamp, tz=timezone.utc) == expected

    def test_format_timestamp_local_timezone(self):
        """Test that the local timezone is used by default."""
        utc = datetime(2026, 8, 13, 12, 26, 2, tzinfo=timezone.utc)
        expected = utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        assert nfl_metadata.format_timestamp(NFLVERSE_TIMESTAMP) == expected

    @pytest.mark.parametrize(
        "timestamp",
        ["2026-08-13 08:26:02 XYZ", "2026-08-13 08:26:02", "13.08.2026 EDT", ""],
    )
    def test_format_timestamp_unknown_format(self, timestamp):
        """Test that unknown formats are returned unchanged."""
        assert nfl_metadata.format_timestamp(timestamp, tz=timezone.utc) == timestamp

    def test_message(self):
        """Test the line for a file with type and timestamp."""
        local = nfl_metadata.format_timestamp(NFLVERSE_TIMESTAMP)
        assert (
            nfl_metadata.metadata_message(NFLVERSE_METADATA)
            == f"nflverse play by play data | Data updated: {local}"
        )

    def test_message_does_not_repeat_nflverse(self):
        """Test types that already start with "nflverse"."""
        message = nfl_metadata.metadata_message(
            {
                "nflverse_type": "nflverse Players Data",
                "nflverse_timestamp": NFLVERSE_TIMESTAMP,
            }
        )
        assert message is not None
        assert message.startswith("nflverse Players Data | Data updated: ")

    def test_message_without_type(self):
        """Test the line for a file that only has a timestamp."""
        message = nfl_metadata.metadata_message(
            {"nflverse_timestamp": NFLVERSE_TIMESTAMP}
        )
        assert message is not None
        assert message.startswith("Data updated: ")

    def test_message_without_timestamp(self):
        """Test that there is no line without a timestamp."""
        assert nfl_metadata.metadata_message({}) is None
        assert nfl_metadata.metadata_message({"nflverse_type": NFLVERSE_TYPE}) is None


class TestDownloaderMetadata:
    """Test metadata handling in the downloader with a fake HTTP session."""

    def test_download_attaches_metadata(self, monkeypatch):
        """Test that the metadata of a downloaded parquet file is available."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        df = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_load_pbp_returns_metadata(self, monkeypatch):
        """Test that metadata is available on the frame load_pbp returns."""
        session = FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        df = nfl.load_pbp(2025)
        assert session.calls == [
            "https://github.com/nflverse/nflverse-data/releases/download/"
            "pbp/play_by_play_2025.parquet"
        ]
        assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_old_file_without_metadata_is_silent(self, monkeypatch, capsys):
        """Test that files saved before the metadata change still load."""
        FakeSession(monkeypatch, make_parquet())
        update_config(show_metadata=True)
        df = get_downloader().download("nflverse-data", "pbp/play_by_play_1999")
        assert len(df) == 3
        assert nfl.get_metadata(df) == {}
        assert capsys.readouterr().out == ""

    def test_csv_file_has_no_metadata(self, monkeypatch, capsys):
        """Test that csv downloads work and have no metadata."""
        FakeSession(monkeypatch, b"season,play_id\n2025,1\n2025,2\n")
        update_config(show_metadata=True)
        df = get_downloader().download(
            "dynastyprocess", "db_playerids", format=DataFormat.CSV
        )
        assert len(df) == 2
        assert nfl.get_metadata(df) == {}
        assert capsys.readouterr().out == ""

    def test_data_updated_line_is_off_by_default(self, monkeypatch, capsys):
        """Test that nothing is printed with the default config."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        assert capsys.readouterr().out == ""

    def test_data_updated_line_with_show_metadata(self, monkeypatch, capsys):
        """Test that show_metadata prints the line without verbose output."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        update_config(show_metadata=True)
        get_downloader().download("nflverse-data", "pbp/play_by_play_2025")

        local = nfl_metadata.format_timestamp(NFLVERSE_TIMESTAMP)
        assert (
            capsys.readouterr().out
            == f"nflverse play by play data | Data updated: {local}\n"
        )

    def test_data_updated_line_follows_verbose(self, monkeypatch, capsys):
        """Test that verbose mode prints the line if show_metadata is not set."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        update_config(verbose=True)
        get_downloader().download("nflverse-data", "pbp/play_by_play_2025")

        output = capsys.readouterr().out
        assert "Downloading" in output
        assert "nflverse play by play data | Data updated: " in output

    def test_data_updated_line_can_be_turned_off(self, monkeypatch, capsys):
        """Test that show_metadata=False hides the line in verbose mode."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        update_config(verbose=True, show_metadata=False)
        get_downloader().download("nflverse-data", "pbp/play_by_play_2025")

        output = capsys.readouterr().out
        assert "Downloading" in output
        assert "Data updated" not in output

    def test_memory_cache_keeps_metadata(self, monkeypatch, capsys):
        """Test that data served from the memory cache has metadata."""
        session = FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        update_config(cache_mode=CacheMode.MEMORY, show_metadata=True)

        first = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        second = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")

        assert len(session.calls) == 1
        assert nfl.get_metadata(first) == NFLVERSE_METADATA
        assert nfl.get_metadata(second) == NFLVERSE_METADATA
        assert capsys.readouterr().out.count("Data updated: ") == 2

    def test_filesystem_cache_keeps_metadata(self, monkeypatch, tmp_path):
        """Test that the metadata survives the parquet cache on disk."""
        session = FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        update_config(cache_mode=CacheMode.FILESYSTEM, cache_dir=tmp_path)

        first = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        second = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")

        assert len(session.calls) == 1
        assert len(list(tmp_path.glob("*.parquet"))) == 1
        assert second is not first
        assert second.equals(first)
        assert nfl.get_metadata(first) == NFLVERSE_METADATA
        assert nfl.get_metadata(second) == NFLVERSE_METADATA

    def test_filesystem_cache_without_metadata(self, monkeypatch, tmp_path):
        """Test the cache on disk for files without metadata."""
        session = FakeSession(monkeypatch, make_parquet())
        update_config(cache_mode=CacheMode.FILESYSTEM, cache_dir=tmp_path)

        get_downloader().download("nflverse-data", "pbp/play_by_play_1999")
        second = get_downloader().download("nflverse-data", "pbp/play_by_play_1999")

        assert len(session.calls) == 1
        assert len(second) == 3
        assert nfl.get_metadata(second) == {}
