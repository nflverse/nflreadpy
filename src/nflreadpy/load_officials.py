"""Load NFL officials data."""

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def load_officials(seasons: int | list[int] | bool | None = True) -> pl.DataFrame:
    """
    Load NFL officials data.

    Data covers NFL officials assigned to games from 2015 onwards.

    Args:
        seasons: Season(s) to load. If True (default), loads all available data.
                If int or list of ints, loads specified season(s).
                If None, loads current season.

    Returns:
        Polars DataFrame with officials data including referee assignments,
        crew information, and game details.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_officials.html>
    """
    downloader = get_downloader()

    # Load the full officials dataset
    df = downloader.download("nflverse-data", "officials/officials")

    # Filter by seasons if specified
    if seasons is not True:
        if seasons is None:
            seasons = [get_current_season()]
        elif isinstance(seasons, int):
            seasons = [seasons]

        # Validate seasons (2015 minimum)
        current_season = get_current_season()
        for season in seasons:
            if not isinstance(season, int) or season < 2015 or season > current_season:
                raise ValueError(f"Season must be between 2015 and {current_season}")

        # Filter the dataframe by season
        if "season" in df.columns:
            df = df.filter(pl.col("season").is_in(seasons))

    return df
