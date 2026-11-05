"""Load NFL team data."""

import polars as pl

from .datasets import team_abbr_mapping
from .downloader import get_downloader


def load_teams(current: bool = False) -> pl.DataFrame:
    """
    Load NFL team information.

    Args:
        current: If True, return only teams with current NFL abbreviations.

    Returns:
        Polars DataFrame with team data including abbreviations, names,\
        colors, logos, and other team metadata.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_teams.html>
    """
    downloader = get_downloader()

    # Load teams data from nflverse-data repository
    df = downloader.download("nflverse-data", "teams/teams_colors_logos")

    if current:
        current_abbrs = team_abbr_mapping().get_column("value").unique().to_list()
        df = df.filter(pl.col("team_abbr").is_in(current_abbrs))

    return df
