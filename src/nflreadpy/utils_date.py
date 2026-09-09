"""Date utility functions for nflreadpy."""

from datetime import date, timedelta

import polars as pl


def _season_start(season_year: int) -> date:
    """
    Get the date a given NFL season begins: the Wednesday following Labor Day.

    Labor Day is the first Monday in September, so this lands two days later.

    This is the season *boundary*, which is not always the date of the first
    game. The 2026 season opens on this Wednesday, but 2015-2025 all opened on
    the Thursday one day later. The boundary sits on the Wednesday because that
    is a no-game day in either case: weeks run Wednesday-to-Tuesday and games
    fall on Thursday through Monday, so week numbering comes out correct for
    Wednesday- and Thursday-opening seasons alike.

    Args:
        season_year: The season year to compute the start date for.

    Returns:
        The date the season begins.
    """
    # Labor Day is the first Monday in September
    for day in range(1, 8):
        labor_day = date(season_year, 9, day)
        if labor_day.weekday() == 0:  # Monday
            break

    # Wednesday following Labor Day
    return labor_day + timedelta(days=2)


def get_current_season(roster: bool = False) -> int:
    """
    Get the current NFL season year.

    Args:
        roster:
            - If True, uses roster year logic (current year after March 15).
            - If False, uses season logic (current year after Wednesday following Labor Day).

    Returns:
        The current season/roster year.

    See Also:
        <https://nflreadr.nflverse.com/reference/get_current_season.html>
    """
    if not isinstance(roster, bool):
        raise TypeError("argument `roster` must be boolean")

    today = date.today()
    current_year = today.year

    if roster:
        # Roster logic: current year after March 15, otherwise previous year
        march_15 = date(current_year, 3, 15)
        return current_year if today >= march_15 else current_year - 1
    else:
        # Season logic: current year after Wednesday following Labor Day
        season_start = _season_start(current_year)
        return current_year if today >= season_start else current_year - 1


def get_current_week(use_date: bool = False, **kwargs) -> int:
    """
    Get the current NFL week (rough approximation).

    Args:
        use_date:
            - If `True`, calculates week as the number of weeks since Wednesday following Labor Day.
            - If `False`, loads schedules via `load_schedules(seasons = get_current_season(**kwargs))` and returns week of the next game.
        **kwargs:
            Arguments passed on to `get_current_season()`

    Returns:
        The current NFL week (1-22).

    See Also:
        <https://nflreadr.nflverse.com/reference/get_current_week.html>
    """
    if not isinstance(use_date, bool):
        raise TypeError("argument `use_date` must be boolean")

    from .load_schedules import load_schedules

    if use_date:
        today = date.today()
        season_year = get_current_season(**kwargs)

        # The NFL season starts on the Wednesday following Labor Day
        season_start = _season_start(season_year)

        if today < season_start:
            return 1

        # Calculate weeks since season start
        days_since_start = (today - season_start).days
        week = min(days_since_start // 7 + 1, 22)  # Cap at week 22

        return int(week)
    else:
        sched = load_schedules(seasons=get_current_season(**kwargs))
        count_na_weeks = sched.select("result").null_count().item()
        if count_na_weeks == 0:
            # no NA values in result, return max(week)
            return sched.select("week").drop_nulls().max().item()
        else:
            # there are NA values in result. Filter table to NA results only,
            # and return min(week)
            return (
                sched.filter(pl.col("result").is_null())
                .select("week")
                .drop_nulls()
                .min()
                .item()
            )


def most_recent_season(roster: bool = False) -> int:
    """
    Alias for get_current_season for compatibility with nflreadr.

    Args:
        roster: If True, uses roster year logic.

    Returns:
        The most recent season/roster year.
    """
    return get_current_season(roster=roster)
