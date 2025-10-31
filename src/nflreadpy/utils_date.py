"""Date utility functions for nflreadpy."""

from datetime import date

import polars as pl

from .load_schedules import load_schedules


def get_current_season(roster: bool = False) -> int:
    """
    Get the current NFL season year.

    Args:
        roster:
            - If True, uses roster year logic (current year after March 15).
            - If False, uses season logic (current year after Thursday following Labor Day).

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
        # Season logic: current year after Thursday following Labor Day
        # Labor Day is first Monday in September
        # Find first Monday in September
        for day in range(1, 8):
            if date(current_year, 9, day).weekday() == 0:  # Monday
                labor_day = date(current_year, 9, day)
                break

        # Thursday following Labor Day
        season_start = date(labor_day.year, labor_day.month, labor_day.day + 3)
        return current_year if today >= season_start else current_year - 1


def get_current_week(use_date: bool = False, **kwargs) -> int:
    """
    Get the current NFL week (rough approximation).

    Args:
        use_date:
            - If `True`, calculates week as the number of weeks since Thursday following Labor Day.
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

    if use_date:
        today = date.today()
        season_year = get_current_season(**kwargs)

        # NFL season typically starts around first Thursday of September
        # Find first Thursday in September
        for day in range(1, 8):
            if date(season_year, 9, day).weekday() == 3:  # Thursday
                season_start = date(season_year, 9, day)
                break

        if today < season_start:
            return 1

        # Calculate weeks since season start
        days_since_start = (today - season_start).days
        week = min(days_since_start // 7 + 1, 22)  # Cap at week 22

        return int(week)
    else:
        # Polars is incredible but come on the syntax is insane
        sched = load_schedules(seasons = get_current_season(**kwargs))
        # counts NA values in column result
        if sched.select(pl.col("result")).null_count().item() == 0:
            # no NA values in result, return max(week)
            return sched.select(pl.col("week")).drop_nulls().max().item()
        else:
            # there are NA values in result. Filter table to NA results only, 
            # and return min(week)
            return sched.filter(pl.col("result").is_null()).select(pl.col("week")).drop_nulls().min().item()


def most_recent_season(roster: bool = False) -> int:
    """
    Alias for get_current_season for compatibility with nflreadr.

    Args:
        roster: If True, uses roster year logic.

    Returns:
        The most recent season/roster year.
    """
    return get_current_season(roster=roster)
