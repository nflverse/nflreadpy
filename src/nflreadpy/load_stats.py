"""Load NFL player and team statistics."""

from typing import Literal

import polars as pl

from .downloader import get_downloader
from .utils_date import get_current_season


def _load_stats(
    stat_type: str,
    seasons: int | list[int] | bool | None = None,
    summary_level: Literal["week", "reg", "post", "reg+post"] = "week",
) -> pl.DataFrame:
    """
    Internal function to load NFL statistics.

    Args:
        stat_type: Type of stats ("player" or "team").
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data.
                If int or list of ints, loads specified season(s).
        summary_level: Summary level ("week", "reg", "post", "reg+post").

    Returns:
        Polars DataFrame with statistics.
    """
    if seasons is None:
        seasons = [get_current_season()]
    elif seasons is True:
        # Load all available seasons
        current_season = get_current_season()
        seasons = list(range(1999, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    if summary_level not in ["week", "reg", "post", "reg+post"]:
        raise ValueError("summary_level must be 'week', 'reg', 'post', or 'reg+post'")

    if stat_type not in ["player", "team"]:
        raise ValueError("stat_type must be 'player' or 'team'")

    # Convert summary level for URL path
    level_str = summary_level.replace("+", "")  # "reg+post" becomes "regpost"

    downloader = get_downloader()
    dataframes = []

    for season in seasons:
        path = f"stats_{stat_type}/stats_{stat_type}_{level_str}_{season}"
        df = downloader.download(
            "nflverse-data",
            path,
            season=season,
            summary_level=summary_level,
            stat_type=stat_type,
        )
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes, how="diagonal_relaxed")


def load_player_stats(
    seasons: int | list[int] | bool | None = None,
    summary_level: Literal["week", "reg", "post", "reg+post"] = "week",
) -> pl.DataFrame:
    """
    Load NFL player statistics.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data.
                If int or list of ints, loads specified season(s).
        summary_level: Summary level ("week", "reg", "post", "reg+post").

    Returns:
        Polars DataFrame with player statistics.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_player_stats.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_player_stats.html>
    """
    return _load_stats("player", seasons, summary_level)


def load_team_stats(
    seasons: int | list[int] | bool | None = None,
    summary_level: Literal["week", "reg", "post", "reg+post"] = "week",
) -> pl.DataFrame:
    """
    Load NFL team statistics.

    Args:
        seasons: Season(s) to load. If None, loads current season.
                If True, loads all available data.
                If int or list of ints, loads specified season(s).
        summary_level: Summary level ("week", "reg", "post", "reg+post").

    Returns:
        Polars DataFrame with team statistics.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_team_stats.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_team_stats.html>
    """
    return _load_stats("team", seasons, summary_level)
