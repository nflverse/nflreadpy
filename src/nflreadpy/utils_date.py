"""Date utility functions for nflreadpy."""

from datetime import date


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


def get_current_week() -> int:
    """
    Get the current NFL week (rough approximation).

    Returns:
        The current NFL week (1-22).

    See Also:
        <https://nflreadr.nflverse.com/reference/get_current_week.html>
    """
    today = date.today()
    season_year = get_current_season()

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


def most_recent_season(roster: bool = False) -> int:
    """
    Alias for get_current_season for compatibility with nflreadr.

    Args:
        roster: If True, uses roster year logic.

    Returns:
        The most recent season/roster year.
    """
    return get_current_season(roster=roster)
