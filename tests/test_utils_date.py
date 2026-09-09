"""Unit tests for date utilities (no network access required)."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from nflreadpy import utils_date
from nflreadpy.utils_date import _season_start, get_current_season, get_current_week


def _frozen(today: date):
    """Patch `date.today()` inside utils_date to return a fixed date."""

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return today

    return patch.object(utils_date, "date", FakeDate)


class TestSeasonStart:
    """
    The season boundary is the Wednesday following Labor Day.

    This is the rollover boundary, not necessarily the date of the first game:
    2026 opens on that Wednesday, but 2015-2025 all opened on the Thursday one
    day later. The boundary is deliberately placed on the Wednesday because it
    falls on a no-game day in either case, so week numbering comes out correct
    for Wednesday- and Thursday-opening seasons alike.
    """

    @pytest.mark.parametrize(
        ("season", "expected"),
        [
            (2023, date(2023, 9, 6)),
            (2024, date(2024, 9, 4)),
            (2025, date(2025, 9, 3)),
            (2026, date(2026, 9, 9)),  # 2026 opener: NE @ SEA
            (2027, date(2027, 9, 8)),
            (2028, date(2028, 9, 6)),
        ],
    )
    def test_known_season_boundaries(self, season, expected):
        assert _season_start(season) == expected

    def test_matches_the_2026_opener(self):
        """2026 is the first season to actually kick off on this boundary."""
        assert _season_start(2026) == date(2026, 9, 9)
        assert _season_start(2026).strftime("%A") == "Wednesday"

    @pytest.mark.parametrize("season", range(1999, 2051))
    def test_always_a_wednesday_after_labor_day(self, season):
        start = _season_start(season)
        assert start.weekday() == 2, "season must start on a Wednesday"
        # Labor Day is the first Monday in September, i.e. two days earlier.
        labor_day = start - timedelta(days=2)
        assert labor_day.month == 9
        assert labor_day.weekday() == 0
        assert labor_day.day <= 7, "Labor Day is the first Monday in September"


class TestGetCurrentSeason:
    """The season rolls over on opening day, not before."""

    def test_day_before_opener_is_previous_season(self):
        with _frozen(date(2026, 9, 8)):  # Tuesday
            assert get_current_season() == 2025

    def test_opening_day_is_new_season(self):
        with _frozen(date(2026, 9, 9)):  # Wednesday opener
            assert get_current_season() == 2026

    def test_midseason(self):
        with _frozen(date(2026, 11, 1)):
            assert get_current_season() == 2026

    def test_roster_year_flips_on_march_15(self):
        with _frozen(date(2026, 3, 14)):
            assert get_current_season(roster=True) == 2025
        with _frozen(date(2026, 3, 15)):
            assert get_current_season(roster=True) == 2026

    def test_rejects_non_boolean(self):
        with pytest.raises(TypeError):
            get_current_season(roster="yes")  # type: ignore[arg-type]


class TestGetCurrentWeekByDate:
    """Weeks are counted from opening day and roll over on Wednesdays."""

    @pytest.mark.parametrize(
        ("today", "expected"),
        [
            (date(2026, 9, 9), 1),  # opening day
            (date(2026, 9, 13), 1),  # Sunday of week 1
            (date(2026, 9, 15), 1),  # Tuesday, still week 1
            (date(2026, 9, 16), 2),  # Wednesday, week 2 begins
            (date(2027, 1, 6), 18),  # final week of the regular season
        ],
    )
    def test_week_boundaries(self, today, expected):
        with _frozen(today):
            assert get_current_week(use_date=True) == expected

    def test_before_the_opener_returns_week_one(self):
        # Preseason: the 2026 season has not started, so we are still in 2025.
        with _frozen(date(2026, 7, 1)):
            assert get_current_week(use_date=True, roster=True) == 1

    def test_week_is_capped_at_22(self):
        with _frozen(date(2027, 6, 1)):
            assert get_current_week(use_date=True) <= 22

    def test_rejects_non_boolean(self):
        with pytest.raises(TypeError):
            get_current_week(use_date="yes")  # type: ignore[arg-type]
