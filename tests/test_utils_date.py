"""Test date utility functions."""

from datetime import date
from unittest.mock import patch

from nflreadpy.utils_date import (
    get_current_season,
    get_current_week,
    most_recent_season,
)


class TestGetCurrentSeason:
    """Test get_current_season function."""

    def test_season_logic_before_labor_day(self):
        """Test season logic returns previous year before Labor Day."""
        # Mock date to August 15, 2024 (before Labor Day)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 8, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            season = get_current_season(roster=False)
            assert season == 2023

    def test_season_logic_after_labor_day(self):
        """Test season logic returns current year after Labor Day."""
        # Mock date to September 15, 2024 (after Labor Day)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 9, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            season = get_current_season(roster=False)
            assert season == 2024

    def test_roster_logic_before_march_15(self):
        """Test roster logic returns previous year before March 15."""
        # Mock date to March 10, 2024 (before March 15)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 3, 10)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            season = get_current_season(roster=True)
            assert season == 2023

    def test_roster_logic_after_march_15(self):
        """Test roster logic returns current year after March 15."""
        # Mock date to March 20, 2024 (after March 15)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 3, 20)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            season = get_current_season(roster=True)
            assert season == 2024


class TestGetCurrentWeek:
    """Test get_current_week function."""

    def test_before_season_start(self):
        """Test week returns 1 before season starts."""
        # Mock date to August 15, 2024 (before season)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 8, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            with patch("nflreadpy.utils_date.get_current_season", return_value=2024):
                week = get_current_week()
                assert week == 1

    def test_during_season(self):
        """Test week calculation during season."""
        # Mock date to September 15, 2024 (during season)
        with patch("nflreadpy.utils_date.date") as mock_date:
            mock_date.today.return_value = date(2024, 9, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)

            with patch("nflreadpy.utils_date.get_current_season", return_value=2024):
                week = get_current_week()
                assert 1 <= week <= 22


def test_most_recent_season_alias():
    """Test most_recent_season is an alias for get_current_season."""
    with patch(
        "nflreadpy.utils_date.get_current_season", return_value=2024
    ) as mock_get:
        result = most_recent_season(roster=True)
        assert result == 2024
        mock_get.assert_called_once_with(roster=True)
