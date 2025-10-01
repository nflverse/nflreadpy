"""Load NFL weekly rosters data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_rosters_weekly(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL weekly rosters data.

    Data available from 2002 onwards.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 2002.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with weekly roster data including player status\
        changes, injury designations, and week-by-week roster moves.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_rosters_weekly.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_roster_status.html>
    """
    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons (2002 to current)
        current_season = get_current_season()
        seasons = list(range(2002, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season()
    for season in seasons:
        if not isinstance(season, int) or season < 2002 or season > current_season:
            raise ValueError(f"Season must be between 2002 and {current_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"weekly_rosters/roster_weekly_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes, how="diagonal_relaxed")
