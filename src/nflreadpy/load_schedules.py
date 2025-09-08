"""Load NFL schedule data."""

import polars as pl

from .config import DataFormat
from .downloader import get_downloader
from .utils_date import get_current_season


def load_schedules(seasons: int | list[int] | bool | None = True) -> pl.DataFrame:
    """
    Load NFL schedules.

    Args:
        seasons: Season(s) to load. If True (default), loads all available data.
                If int or list of ints, loads specified season(s).
                If None, loads current season.

    Returns:
        Polars DataFrame with schedule data.
    """
    downloader = get_downloader()

    # Load the full games dataset (default to CSV since we can't parse RDS)
    df = downloader.download("nfldata", "games", format_preference=DataFormat.CSV)

    # Filter by seasons if specified
    if seasons is not True:
        if seasons is None:
            seasons = [get_current_season()]
        elif isinstance(seasons, int):
            seasons = [seasons]

        # Filter the dataframe by season
        df = df.filter(pl.col("season").is_in(seasons))

    # Validate and clean roof values (matching nflreadr logic)
    if "roof" in df.columns:
        valid_roof_values = ["dome", "outdoors", "closed", "open"]
        df = df.with_columns(
            pl.when(pl.col("roof").is_in(valid_roof_values))
            .then(pl.col("roof"))
            .otherwise(None)
            .alias("roof")
        )

    return df
