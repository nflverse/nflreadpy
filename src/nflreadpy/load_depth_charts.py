"""Load NFL depth charts data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_depth_charts(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL depth charts data.

    Data available from 2001 onwards.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 2001.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with depth charts data including player positions,
        depth chart rankings, and team information.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_depth_charts.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html>
    """
    if seasons is None:
        seasons = [get_current_season(roster=True)]
    elif seasons is True:
        # Load all available seasons (2001 to current)
        current_season = get_current_season(roster=True)
        seasons = list(range(2001, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season(roster=True)
    for season in seasons:
        if not isinstance(season, int) or season < 2001 or season > current_season:
            raise ValueError(f"Season must be between 2001 and {current_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"depth_charts/depth_charts_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
