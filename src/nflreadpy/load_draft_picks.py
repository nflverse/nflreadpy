"""Load NFL draft pick data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_draft_picks(seasons: int | list[int] | bool | None = True) -> pl.DataFrame:
    """
    Load NFL draft pick data.

    Data covers draft picks since 1980, sourced from Pro Football Reference.

    Args:
        seasons: Season(s) to load. If True (default), loads all available data.
                If int or list of ints, loads specified season(s).
                If None, loads current season.

    Returns:
        Polars DataFrame with draft pick data including draft year, round,
        pick number, player information, and team data.
    """
    downloader = get_downloader()

    # Load the full draft picks dataset
    df = downloader.download("nflverse-data", "draft_picks/draft_picks")

    # Filter by seasons if specified
    if seasons is not True:
        if seasons is None:
            seasons = [get_current_season()]
        elif isinstance(seasons, int):
            seasons = [seasons]

        # Filter the dataframe by season
        df = df.filter(pl.col("season").is_in(seasons))

    return df
