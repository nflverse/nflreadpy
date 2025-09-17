"""Integration tests for all nflreadpy functions."""

import polars as pl
import pytest

import nflreadpy as nfl


class TestImports:
    """Test that all functions can be imported successfully."""

    def test_all_exports(self):
        """Test that all expected exports are available."""
        expected_exports = [
            # Core loading functions
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
            "load_ftn_charting",
            "load_rosters_weekly",
            # ffverse functions
            "load_ff_playerids",
            "load_ff_rankings",
            "load_ff_opportunity",
            # Utility functions
            "get_current_season",
            "get_current_week",
            "clear_cache",
        ]

        for export in expected_exports:
            assert hasattr(nfl, export), f"Missing export: {export}"
            assert callable(getattr(nfl, export)), f"Export is not callable: {export}"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_current_season(self):
        """Test get_current_season function."""
        season = nfl.get_current_season()
        assert isinstance(season, int)
        assert 2025 <= season <= 2100  # Reasonable bounds

    def test_get_current_week(self):
        """Test get_current_week function."""
        week = nfl.get_current_week()
        assert isinstance(week, int)
        assert 1 <= week <= 22  # Reasonable bounds for NFL weeks

    def test_clear_cache(self):
        """Test clear_cache function."""
        # Should not raise an exception
        nfl.clear_cache()


class TestStaticDataLoaders:
    """Test loaders that don't require season parameters."""

    def test_load_teams(self):
        """Test load_teams function."""
        df = nfl.load_teams()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # Should have 32+ teams (accounting for relocations)
        assert len(df) >= 32

    def test_load_players(self):
        """Test load_players function."""
        df = nfl.load_players()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_trades(self):
        """Test load_trades function."""
        df = nfl.load_trades()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_contracts(self):
        """Test load_contracts function."""
        df = nfl.load_contracts()
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_playerids(self):
        """Test load_ff_playerids function."""
        df = nfl.load_ff_playerids()
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_rankings_draft(self):
        """Test load_ff_rankings with draft type."""
        df = nfl.load_ff_rankings("draft")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_rankings_week(self):
        """Test load_ff_rankings with week type."""
        df = nfl.load_ff_rankings("week")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_rankings_all(self):
        """Test load_ff_rankings with all type."""
        df = nfl.load_ff_rankings("all")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0


class TestSeasonalDataLoaders:
    """Test loaders that require season parameters."""

    def test_load_pbp_2024_season(self):
        """Test load_pbp with 2024 season."""
        df = nfl.load_pbp(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_rosters_2024_season(self):
        """Test load_rosters with 2024 season."""
        df = nfl.load_rosters(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_schedules_2024_season(self):
        """Test load_schedules with 2024 season."""
        df = nfl.load_schedules(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_player_stats_2024_season(self):
        """Test load_player_stats with 2024 season."""
        df = nfl.load_player_stats(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_player_stats_multiple_seasons(self):
        """Test load_pbp with multiple seasons."""
        df = nfl.load_player_stats([2022, 2023])
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_team_stats_2024_season(self):
        """Test load_team_stats with 2024 season."""
        df = nfl.load_team_stats(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_injuries_specific_season(self):
        """Test load_injuries with current season."""
        df = nfl.load_injuries(2023)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_depth_charts_2024_season(self):
        """Test load_depth_charts with 2024 season."""
        df = nfl.load_depth_charts(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_snap_counts_2024_season(self):
        """Test load_snap_counts with 2024 season."""
        df = nfl.load_snap_counts(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_nextgen_stats_2024_season(self):
        """Test load_nextgen_stats with 2024 season."""
        df = nfl.load_nextgen_stats(2024, "passing")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_officials_2024_season(self):
        """Test load_officials with 2024 season."""
        df = nfl.load_officials(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_participation_2024_season(self):
        """Test load_participation with 2024 season."""
        df = nfl.load_participation(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_draft_picks_2024_season(self):
        """Test load_draft_picks with 2024 season."""
        df = nfl.load_draft_picks(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ftn_charting_2024_season(self):
        """Test load_ftn_charting with 2024 season."""
        df = nfl.load_ftn_charting(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_rosters_weekly_2024_season(self):
        """Test load_rosters_weekly with 2024 season."""
        df = nfl.load_rosters_weekly(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_combine_2024_season(self):
        """Test load_combine with 2024 season."""
        df = nfl.load_combine(2024)
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_opportunity_2024_season_week(self):
        """Test load_ff_opportunity week with specific season."""
        df = nfl.load_ff_opportunity(2024, stat_type="weekly")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_opportunity_2024_season_pbp_rush(self):
        """Test load_ff_opportunity pbp_rush with specific season."""
        df = nfl.load_ff_opportunity(2024, stat_type="pbp_rush")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0

    def test_load_ff_opportunity_2024_season_pbp_pass(self):
        """Test load_ff_opportunity pbp_pass with specific season."""
        df = nfl.load_ff_opportunity(2024, stat_type="pbp_pass")
        assert isinstance(df, pl.DataFrame)
        assert len(df) >= 0


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_load_pbp_invalid_season(self):
        """Test load_pbp with invalid season."""
        with pytest.raises(ValueError):
            nfl.load_pbp(1990)  # Too early

        with pytest.raises(ValueError):
            nfl.load_pbp(2100)  # Too far in future

    def test_load_pbp_invalid_type(self):
        """Test load_pbp with invalid type."""
        with pytest.raises((ValueError, TypeError)):
            nfl.load_pbp("invalid")

    def test_load_ff_rankings_invalid_type(self):
        """Test load_ff_rankings with invalid type."""
        with pytest.raises(ValueError):
            nfl.load_ff_rankings("invalid")

    def test_load_ff_opportunity_invalid_season(self):
        """Test load_ff_opportunity with invalid season."""
        with pytest.raises(ValueError):
            nfl.load_ff_opportunity(2005)  # Too early

    def test_load_ff_opportunity_invalid_stat_type(self):
        """Test load_ff_opportunity with invalid stat_type."""
        with pytest.raises(ValueError):
            nfl.load_ff_opportunity(2023, stat_type="invalid")

    def test_load_ff_opportunity_invalid_model_version(self):
        """Test load_ff_opportunity with invalid model_version."""
        with pytest.raises(ValueError):
            nfl.load_ff_opportunity(2023, model_version="invalid")


class TestDataQuality:
    """Test basic data quality for a subset of functions."""

    def test_teams_data_structure(self):
        """Test that teams data has expected structure."""
        df = nfl.load_teams()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # Should have some basic columns (exact names may vary)
        assert len(df.columns) > 5

    def test_pbp_data_structure(self):
        """Test that PBP data has expected structure for a recent season."""
        df = nfl.load_pbp(2023)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # Should have many columns for PBP data
        assert len(df.columns) > 50

    def test_schedules_data_structure(self):
        """Test that schedules data has expected structure."""
        df = nfl.load_schedules(2023)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # Should have ~272 games per season (17 weeks * 16 games + playoffs)
        assert len(df) > 250
