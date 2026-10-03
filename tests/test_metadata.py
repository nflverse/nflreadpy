"""Tests for nflverse metadata, without network access."""

import gc
import io
from datetime import datetime, timedelta, timezone

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
SCHEDULE_METADATA = {
    "nflverse_type": "games and schedules",
    "nflverse_timestamp": "2026-10-03 00:36:18 EDT",
}


def make_parquet(metadata=None, season=2025, df=None):
    """Create a small parquet file in memory, optionally with key-value metadata."""
    if metadata is not None and not hasattr(pl, "read_parquet_metadata"):
        pytest.skip("Polars version has no parquet key-value metadata support")

    if df is None:
        df = pl.DataFrame(
            {
                "season": [season, season, season],
                "play_id": [1, 2, 3],
                "roof": ["dome", "outdoors", "unknown"],
            }
        )
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
    """Serves one file, or a dict of file name to content, and records the URLs."""

    def __init__(self, monkeypatch, content):
        self.content = content
        self.calls = []
        monkeypatch.setattr(get_downloader().session, "get", self.get)

    def get(self, url, **kwargs):
        self.calls.append(url)
        if isinstance(self.content, dict):
            return FakeResponse(self.content[url.rsplit("/", 1)[-1]])
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

    def test_copy_metadata(self):
        """Test copying metadata to a frame built from another frame."""
        source = pl.DataFrame({"a": [1, 2]})
        nfl_metadata.attach_metadata(source, NFLVERSE_METADATA)

        target = source.filter(pl.col("a") > 1)
        assert nfl_metadata.copy_metadata(source, target) is target
        assert nfl.get_metadata(target) == NFLVERSE_METADATA

    def test_copy_metadata_from_frame_without_metadata(self):
        """Test that copying from a frame without metadata leaves none."""
        target = pl.DataFrame({"a": [1]})
        nfl_metadata.attach_metadata(target, NFLVERSE_METADATA)
        nfl_metadata.copy_metadata(pl.DataFrame({"a": [1]}), target)
        assert nfl.get_metadata(target) == {}

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
        local = datetime(2026, 8, 13, 12, 26, 2, tzinfo=timezone.utc).astimezone()
        stamp = local.strftime("%Y-%m-%d %H:%M:%S")
        # Long names (Windows) become a UTC offset
        assert nfl_metadata.format_timestamp(NFLVERSE_TIMESTAMP) in {
            f"{stamp} {local.tzname()}",
            f"{stamp} UTC{local:%z}",
        }

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("CEST", "2026-08-13 14:26:02 CEST"),
            ("W. Europe Daylight Time", "2026-08-13 14:26:02 UTC+0200"),
            ("Mitteleurop\u00e4ische Sommerzeit", "2026-08-13 14:26:02 UTC+0200"),
        ],
    )
    def test_format_timestamp_long_timezone_name(self, name, expected):
        """Test that long or non-ASCII timezone names become a UTC offset."""
        tz = timezone(timedelta(hours=2), name)
        assert nfl_metadata.format_timestamp(NFLVERSE_TIMESTAMP, tz=tz) == expected

    def test_format_timestamp_out_of_range(self):
        """Test that a timestamp that cannot be converted is returned unchanged."""
        timestamp = "9999-12-31 23:59:59 EST"
        assert nfl_metadata.format_timestamp(timestamp, tz=timezone.utc) == timestamp

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

    def test_out_of_range_timestamp_does_not_fail_the_load(self, monkeypatch, capsys):
        """Test that a timestamp that cannot be converted never fails a load."""
        metadata = {
            **NFLVERSE_METADATA,
            "nflverse_timestamp": "9999-12-31 23:59:59 EST",
        }
        FakeSession(monkeypatch, make_parquet(metadata))
        update_config(show_metadata=True)
        df = get_downloader().download("nflverse-data", "pbp/play_by_play_2025")
        assert len(df) == 3
        assert nfl.get_metadata(df) == metadata
        assert "Data updated: " in capsys.readouterr().out

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


class TestLoaderMetadata:
    """Test that load functions keep metadata when they build a new frame."""

    def test_multiple_seasons_use_last_file(self, monkeypatch):
        """Test that combined seasons carry the metadata of the last file."""
        older = {**NFLVERSE_METADATA, "nflverse_timestamp": "2026-08-13 08:26:18 EDT"}
        FakeSession(
            monkeypatch,
            {
                "play_by_play_2024.parquet": make_parquet(older, season=2024),
                "play_by_play_2025.parquet": make_parquet(NFLVERSE_METADATA),
            },
        )
        df = nfl.load_pbp([2024, 2025])
        assert len(df) == 6
        assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_season_filter_keeps_metadata(self, monkeypatch):
        """Test loaders that filter one file by season."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        df = nfl.load_draft_picks(2025)
        assert len(df) == 3
        assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_load_schedules_keeps_metadata(self, monkeypatch):
        """Test load_schedules, which filters and cleans the roof column."""
        FakeSession(monkeypatch, make_parquet(NFLVERSE_METADATA))
        for df in [nfl.load_schedules(), nfl.load_schedules(2025)]:
            assert df["roof"].to_list() == ["dome", "outdoors", None]
            assert nfl.get_metadata(df) == NFLVERSE_METADATA

    def test_multiple_seasons_without_metadata(self, monkeypatch):
        """Test that combined files without metadata still load."""
        FakeSession(monkeypatch, make_parquet())
        df = nfl.load_pbp([2024, 2025])
        assert len(df) == 6
        assert nfl.get_metadata(df) == {}


class TestInternalDownloads:
    """Test that internal downloads print no line."""

    @staticmethod
    def make_files():
        """Schedule with week 1 played and week 2 open, and a participation file."""
        season = nfl.get_current_season()
        schedule = pl.DataFrame(
            {"season": [season, season], "week": [1, 2], "result": [3, None]}
        )
        return {
            "games.parquet": make_parquet(SCHEDULE_METADATA, df=schedule),
            "pbp_participation_2024.parquet": make_parquet(season=2024),
        }

    def test_get_current_week_is_quiet(self, monkeypatch, capsys):
        """Test that the schedule download in get_current_week() prints no line."""
        FakeSession(monkeypatch, self.make_files())
        update_config(show_metadata=True)
        assert nfl.get_current_week() == 2
        assert capsys.readouterr().out == ""

        # Loading the schedule directly still prints the line
        nfl.load_schedules()
        output = capsys.readouterr().out
        assert "nflverse games and schedules | Data updated: " in output

    def test_get_current_week_is_quiet_in_verbose_mode(self, monkeypatch, capsys):
        """Test that verbose mode prints the download but not the line."""
        FakeSession(monkeypatch, self.make_files())
        update_config(verbose=True)
        assert nfl.get_current_week() == 2

        output = capsys.readouterr().out
        assert "Downloading" in output
        assert "Data updated" not in output

    def test_load_participation_has_no_schedule_line(self, monkeypatch, capsys):
        """Test that load_participation() prints no line for the schedule file."""
        FakeSession(monkeypatch, self.make_files())
        update_config(show_metadata=True)
        df = nfl.load_participation(2024)
        assert len(df) == 3
        assert nfl.get_metadata(df) == {}
        assert capsys.readouterr().out == ""
