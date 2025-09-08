"""Integration tests for nflreadpy."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import nflreadpy as nfl


@pytest.fixture
def mock_dataframe():
    """Create a mock dataframe for testing."""
    return pl.DataFrame({"season": [2023, 2023], "week": [1, 2], "team": ["KC", "BUF"]})


class TestLoadFunctions:
    """Test main load functions."""

    def test_load_pbp_import(self):
        """Test that load_pbp can be imported and called."""
        assert hasattr(nfl, "load_pbp")
        assert callable(nfl.load_pbp)

    def test_load_player_stats_import(self):
        """Test that load_player_stats can be imported and called."""
        assert hasattr(nfl, "load_player_stats")
        assert callable(nfl.load_player_stats)

    def test_load_team_stats_import(self):
        """Test that load_team_stats can be imported and called."""
        assert hasattr(nfl, "load_team_stats")
        assert callable(nfl.load_team_stats)

    def test_load_rosters_import(self):
        """Test that load_rosters can be imported and called."""
        assert hasattr(nfl, "load_rosters")
        assert callable(nfl.load_rosters)

    def test_load_schedules_import(self):
        """Test that load_schedules can be imported and called."""
        assert hasattr(nfl, "load_schedules")
        assert callable(nfl.load_schedules)

    def test_load_teams_import(self):
        """Test that load_teams can be imported and called."""
        assert hasattr(nfl, "load_teams")
        assert callable(nfl.load_teams)

    def test_load_players_import(self):
        """Test that load_players can be imported and called."""
        assert hasattr(nfl, "load_players")
        assert callable(nfl.load_players)

    def test_load_draft_picks_import(self):
        """Test that load_draft_picks can be imported and called."""
        assert hasattr(nfl, "load_draft_picks")
        assert callable(nfl.load_draft_picks)

    def test_load_injuries_import(self):
        """Test that load_injuries can be imported and called."""
        assert hasattr(nfl, "load_injuries")
        assert callable(nfl.load_injuries)

    def test_load_contracts_import(self):
        """Test that load_contracts can be imported and called."""
        assert hasattr(nfl, "load_contracts")
        assert callable(nfl.load_contracts)

    def test_load_snap_counts_import(self):
        """Test that load_snap_counts can be imported and called."""
        assert hasattr(nfl, "load_snap_counts")
        assert callable(nfl.load_snap_counts)

    def test_load_nextgen_stats_import(self):
        """Test that load_nextgen_stats can be imported and called."""
        assert hasattr(nfl, "load_nextgen_stats")
        assert callable(nfl.load_nextgen_stats)

    def test_load_officials_import(self):
        """Test that load_officials can be imported and called."""
        assert hasattr(nfl, "load_officials")
        assert callable(nfl.load_officials)

    def test_load_participation_import(self):
        """Test that load_participation can be imported and called."""
        assert hasattr(nfl, "load_participation")
        assert callable(nfl.load_participation)

    def test_load_combine_import(self):
        """Test that load_combine can be imported and called."""
        assert hasattr(nfl, "load_combine")
        assert callable(nfl.load_combine)

    def test_load_depth_charts_import(self):
        """Test that load_depth_charts can be imported and called."""
        assert hasattr(nfl, "load_depth_charts")
        assert callable(nfl.load_depth_charts)

    def test_load_trades_import(self):
        """Test that load_trades can be imported and called."""
        assert hasattr(nfl, "load_trades")
        assert callable(nfl.load_trades)

    @patch("nflreadpy.load_pbp.get_downloader")
    def test_load_pbp_with_mock(self, mock_get_downloader, mock_dataframe):
        """Test load_pbp with mocked downloader."""
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = mock_dataframe
        mock_get_downloader.return_value = mock_downloader

        with patch("nflreadpy.load_pbp.get_current_season", return_value=2023):
            result = nfl.load_pbp(seasons=2023)

            assert isinstance(result, pl.DataFrame)
            mock_downloader.download.assert_called_once()

    @patch("nflreadpy.load_stats.get_downloader")
    def test_load_player_stats_with_mock(self, mock_get_downloader, mock_dataframe):
        """Test load_player_stats with mocked downloader."""
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = mock_dataframe
        mock_get_downloader.return_value = mock_downloader

        with patch("nflreadpy.load_stats.get_current_season", return_value=2023):
            result = nfl.load_player_stats(seasons=2023)

            assert isinstance(result, pl.DataFrame)
            mock_downloader.download.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_current_season_import(self):
        """Test that get_current_season can be imported and called."""
        assert hasattr(nfl, "get_current_season")
        assert callable(nfl.get_current_season)

        # Should return an integer year
        season = nfl.get_current_season()
        assert isinstance(season, int)
        assert 2020 <= season <= 2100  # Reasonable range

    def test_get_current_week_import(self):
        """Test that get_current_week can be imported and called."""
        assert hasattr(nfl, "get_current_week")
        assert callable(nfl.get_current_week)

        # Should return an integer week
        week = nfl.get_current_week()
        assert isinstance(week, int)
        assert 1 <= week <= 22

    def test_clear_cache_import(self):
        """Test that clear_cache can be imported and called."""
        assert hasattr(nfl, "clear_cache")
        assert callable(nfl.clear_cache)

        # Should not raise an error
        nfl.clear_cache()


def test_package_version():
    """Test package version is accessible."""
    assert hasattr(nfl, "__version__")
    assert isinstance(nfl.__version__, str)


def test_all_exports():
    """Test that __all__ contains expected exports."""
    expected_exports = [
        "load_pbp",
        "load_player_stats",
        "load_team_stats",
        "load_rosters",
        "load_schedules",
        "load_teams",
        "load_players",
        "load_draft_picks",
        "load_injuries",
        "load_contracts",
        "load_snap_counts",
        "load_nextgen_stats",
        "load_officials",
        "load_participation",
        "load_combine",
        "load_depth_charts",
        "load_trades",
        "get_current_season",
        "get_current_week",
        "clear_cache",
    ]

    for export in expected_exports:
        assert export in nfl.__all__
        assert hasattr(nfl, export)
