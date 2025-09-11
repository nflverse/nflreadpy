"""Load NFL roster data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_rosters(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL team rosters.

    Args:
        seasons: Season(s) to load. If None, loads current roster year.
                If True, loads all available data since 1920.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with roster data.

    See Also:
        https://nflreadr.nflverse.com/reference/load_rosters.html

    Data Dictionary:
        https://nflreadr.nflverse.com/articles/dictionary_rosters.html
    """
    if seasons is None:
        seasons = [get_current_season(roster=True)]
    elif seasons is True:
        # Load all available seasons (1920 to current roster year)
        current_roster_year = get_current_season(roster=True)
        seasons = list(range(1920, current_roster_year + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_roster_year = get_current_season(roster=True)
    for season in seasons:
        if not isinstance(season, int) or season < 1920 or season > current_roster_year:
            raise ValueError(f"Season must be between 1920 and {current_roster_year}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"rosters/roster_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
