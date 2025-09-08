"""Load NFL play-by-play data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_pbp(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL play-by-play data.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 1999.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with play-by-play data.
    """
    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons (1999 to current)
        current_season = get_current_season()
        seasons = list(range(1999, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season()
    for season in seasons:
        if not isinstance(season, int) or season < 1999 or season > current_season:
            raise ValueError(f"Season must be between 1999 and {current_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"pbp/play_by_play_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
