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
    """
    if seasons is None:
        seasons = [get_current_season() - 1]
    elif seasons is True:
        # Load all available seasons (2016 to current)
        current_season = get_current_season() - 1
        seasons = list(range(2016, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons - currently only available on historical basis
    current_season = get_current_season() - 1
    for season in seasons:
        if not isinstance(season, int) or season < 2016 or season > current_season:
            raise ValueError(f"Season must be between 2016 and {current_season}")

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"pbp_participation/pbp_participation_{season}"
        df = downloader.download("nflverse-data", path, season=season)
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
