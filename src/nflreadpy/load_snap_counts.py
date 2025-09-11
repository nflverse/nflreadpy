"""Load NFL snap count data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_snap_counts(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL snap count data.

    Data sourced from Pro Football Reference, available since 2012.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 2012.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with snap count data including player information,
        offensive/defensive snaps, and snap percentages.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_snap_counts.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html>
    """
    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons (2012 to current)
        current_season = get_current_season()
        seasons = list(range(2012, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season()
    for season in seasons:
        if not isinstance(season, int) or season < 2012 or season > current_season:
            raise ValueError(f"Season must be between 2012 and {current_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"snap_counts/snap_counts_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
