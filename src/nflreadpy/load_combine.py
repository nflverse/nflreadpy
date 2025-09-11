"""Load NFL Combine data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_combine(seasons: int | list[int] | bool | None = True) -> pl.DataFrame:
    """
    Load NFL Combine data.

    Args:
        seasons: Season(s) to load. If True (default), loads all available data.
                If int or list of ints, loads specified season(s).
                If None, loads current season.

    Returns:
        Polars DataFrame with NFL Combine data including player measurements,
        test results (40-yard dash, bench press, etc.), and draft information.

    See Also:
        https://nflreadr.nflverse.com/reference/load_combine.html
    """
    downloader = get_downloader()

    # Load the full combine dataset
    df = downloader.download("nflverse-data", "combine/combine")

    # Filter by seasons if specified
    if seasons is not True:
        if seasons is None:
            seasons = [get_current_season()]
        elif isinstance(seasons, int):
            seasons = [seasons]

        # Filter the dataframe by season
        if "season" in df.columns:
            df = df.filter(pl.col("season").is_in(seasons))

    return df
