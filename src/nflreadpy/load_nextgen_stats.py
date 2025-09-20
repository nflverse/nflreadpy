"""Load NFL Next Gen Stats data."""

import polars as pl
from typing import Literal
from .downloader import get_downloader
from .utils_date import get_current_season


def load_nextgen_stats(
    seasons: int | list[int] | bool | None = None,
    stat_type: Literal["passing", "receiving", "rushing"] = "passing"
) -> pl.DataFrame:
    """
    Load NFL Next Gen Stats data.

    Data available since 2016.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data since 2016.
                If int or list of ints, loads specified season(s).
        stat_type: Type of stats to load. Options: "passing", "receiving", "rushing".

    Returns:
        Polars DataFrame with Next Gen Stats data including advanced metrics
        for passing, receiving, or rushing performance.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_nextgen_stats.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html>
    """
    if stat_type not in ["passing", "receiving", "rushing"]:
        raise ValueError("stat_type must be 'passing', 'receiving', or 'rushing'")

    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons (2016 to current)
        current_season = get_current_season()
        seasons = list(range(2016, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season()
    for season in seasons:
        if not isinstance(season, int) or season < 2016 or season > current_season:
            raise ValueError(f"Season must be between 2016 and {current_season}")

    downloader = get_downloader()

    # Load the full dataset for the stat type
    path = f"nextgen_stats/ngs_{stat_type}"
    df = downloader.download("nflverse-data", path, stat_type=stat_type)

    # Filter by seasons
    if "season" in df.columns:
        df = df.filter(pl.col("season").is_in(seasons))

    return df
