"""Load Pro Football Reference advanced statistics."""

from typing import Literal

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def _load_pfr_advstats_week(
    seasons: list[int],
    stat_type: Literal["pass", "rush", "rec", "def"],
) -> pl.DataFrame:
    """
    Load weekly Pro Football Reference advanced statistics.

    Args:
        seasons: List of seasons to load.
        stat_type: Type of statistics to load.

    Returns:
        Polars DataFrame with weekly advanced statistics.
    """
    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"pfr_advstats/advstats_week_{stat_type}_{season}"
        df = downloader.download(
            "nflverse-data",
            path,
            season=season,
            stat_type=stat_type,
            summary_level="week",
        )
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes, how="diagonal_relaxed")


def _load_pfr_advstats_season(
    stat_type: Literal["pass", "rush", "rec", "def"],
) -> pl.DataFrame:
    """
    Load season-level Pro Football Reference advanced statistics.

    Args:
        stat_type: Type of statistics to load.

    Returns:
        Polars DataFrame with season-level advanced statistics.
    """
    downloader = get_downloader()
    path = f"pfr_advstats/advstats_season_{stat_type}"
    return downloader.download(
        "nflverse-data",
        path,
        stat_type=stat_type,
        summary_level="season",
    )


def load_pfr_advstats(
    seasons: int | list[int] | bool | None = None,
    stat_type: Literal["pass", "rush", "rec", "def"] = "pass",
    summary_level: Literal["week", "season"] = "week",
) -> pl.DataFrame:
    """
    Load Pro Football Reference advanced statistics.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data (2018-current).
                If int or list of ints, loads specified season(s).
                Only used when summary_level="week".
        stat_type: Type of statistics to load:
                  - "pass": Passing statistics
                  - "rush": Rushing statistics
                  - "rec": Receiving statistics
                  - "def": Defensive statistics
        summary_level: Summary level:
                      - "week": Weekly statistics by season
                      - "season": Season-level statistics (all seasons combined)

    Returns:
        Polars DataFrame with Pro Football Reference advanced statistics.

    Note:
        Data is available from 2018 onwards.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_pfr_advstats.html>
    """
    # Validate stat_type
    if stat_type not in ["pass", "rush", "rec", "def"]:
        raise ValueError("stat_type must be 'pass', 'rush', 'rec', or 'def'")

    # Validate summary_level
    if summary_level not in ["week", "season"]:
        raise ValueError("summary_level must be 'week' or 'season'")

    # For season-level data, ignore seasons parameter
    if summary_level == "season":
        return _load_pfr_advstats_season(stat_type)

    # Handle seasons parameter for week-level data
    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons (2018-current)
        current_season = get_current_season()
        seasons = list(range(2018, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate seasons
    current_season = get_current_season()
    for season in seasons:
        if not isinstance(season, int) or season < 2018 or season > current_season:
            raise ValueError(f"Season must be between 2018 and {current_season}")

    return _load_pfr_advstats_week(seasons, stat_type)
