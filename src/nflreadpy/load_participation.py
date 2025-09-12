"""Load NFL participation data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_participation(seasons: int | list[int] | bool | None = None) -> pl.DataFrame:
    """
    Load NFL participation data.

    Data available since 2016.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 2016.
                If int or list of ints, loads specified season(s).

    Returns:
        Polars DataFrame with participation data including player involvement
        on specific plays and snap participation details.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_participation.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_participation.html>
    """
    # participation only available on a historical basis from FTN
    max_season = get_current_season(roster=True) - 1
    if seasons is None:
        seasons = [max_season]
    elif seasons is True:
        # Load all available seasons (2016 to max_season)
        seasons = list(range(2016, max_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    for season in seasons:
        if not isinstance(season, int) or season < 2016 or season > max_season:
            raise ValueError(f"Season must be between 2016 and {max_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"pbp_participation/pbp_participation_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes, how="diagonal_relaxed")
