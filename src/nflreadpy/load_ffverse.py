"""Load fantasy football data from ffverse."""

from typing import Literal

import polars as pl

from .config import DataFormat
from .downloader import get_downloader
from .utils_date import get_current_season


def load_ff_playerids() -> pl.DataFrame:
    """
    Load fantasy football player IDs from DynastyProcess.com database.

    Returns:
        Polars DataFrame with comprehensive player ID mappings across platforms.

    Note:
        This function loads data from an R data file (.rds). While Python cannot
        directly read RDS files, we attempt to use CSV format if available.

    See Also:
        https://nflreadr.nflverse.com/reference/load_ff_playerids.html
    """
    downloader = get_downloader()

    df = downloader.download(
        "dynastyprocess", "db_playerids", format_preference=DataFormat.CSV
    )

    return df


def load_ff_rankings(type: Literal["draft", "week", "all"] = "draft") -> pl.DataFrame:
    """
    Load fantasy football rankings and projections.

    Args:
        type: Type of rankings to load:
            - "draft": Draft rankings/projections
            - "week": Weekly rankings/projections
            - "all": All historical rankings/projections

    Returns:
        Polars DataFrame with fantasy football rankings data.

    See Also:
        https://nflreadr.nflverse.com/reference/load_ff_rankings.html
    """
    downloader = get_downloader()

    # Map ranking types to file names
    file_mapping = {
        "draft": "db_fpecr_latest",
        "week": "fp_latest_weekly",
        "all": "db_fpecr",
    }

    if type not in file_mapping:
        raise ValueError(f"Invalid type '{type}'. Must be one of: draft, week, all")

    filename = file_mapping[type]

    if type == "all":
        df = downloader.download("dynastyprocess", filename)
    else:
        df = downloader.download(
            "dynastyprocess", filename, format_preference=DataFormat.CSV
        )

    return df


def load_ff_opportunity(
    seasons: int | list[int] | None = None,
    stat_type: Literal["weekly", "pbp_pass", "pbp_rush"] = "weekly",
    model_version: Literal["latest", "v1.0.0"] = "latest",
) -> pl.DataFrame:
    """
    Load fantasy football opportunity data.

    This function loads opportunity and target share data for fantasy football
    analysis from the ffverse/ffopportunity repository.

    Args:
        seasons: Season(s) to load. If None (default), loads current season.
                If int or list of ints, loads specified season(s). True loads all seasons.
        stat_type: Type of stats to load:
            - "weekly": Weekly opportunity data
            - "pbp_pass": Play-by-play passing data
            - "pbp_rush": Play-by-play rushing data
        model_version: Model version to load:
            - "latest": Most recent model version
            - "v1.0.0": Specific model version

    Returns:
        Polars DataFrame with fantasy football opportunity data.

    Raises:
        ValueError: If season is outside valid range or invalid parameters provided.

    See Also:
        https://nflreadr.nflverse.com/reference/load_ff_opportunity.html
    """
    downloader = get_downloader()

    # Validate parameters
    valid_stat_types = ["weekly", "pbp_pass", "pbp_rush"]
    if stat_type not in valid_stat_types:
        raise ValueError(
            f"Invalid stat_type '{stat_type}'. Must be one of: {valid_stat_types}"
        )

    valid_versions = ["latest", "v1.0.0"]
    if model_version not in valid_versions:
        raise ValueError(
            f"Invalid model_version '{model_version}'. Must be one of: {valid_versions}"
        )

    min_year = 2006
    current_season = get_current_season()
    # Handle seasons parameter
    if seasons is None:
        seasons = [current_season]
    elif seasons is True:
        # Load all available seasons (min_year to current)
        current_season = get_current_season()
        seasons = list(range(min_year, current_season + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]

    # Validate season range
    for season in seasons:
        if not isinstance(season, int) or season < min_year or season > current_season:
            raise ValueError(f"Season must be between {min_year} and {current_season}")

    # Load data for each season
    dataframes = []
    for season in seasons:
        # Build the release tag and filename based on the R implementation
        release_tag = f"{model_version}-data"
        filename = f"ep_{stat_type}_{season}"

        # Build the path for the ffopportunity repository
        path = f"{release_tag}/{filename}"

        df = downloader.download("ffopportunity", path)

        dataframes.append(df)

    # Combine all seasons
    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pl.concat(dataframes)
