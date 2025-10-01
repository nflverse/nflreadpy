"""Load NFL team data."""

import polars as pl

from .config import DataFormat
from .downloader import get_downloader


def load_teams() -> pl.DataFrame:
    """
    Load NFL team information.

    Returns:
        Polars DataFrame with team data including abbreviations, names,
        colors, logos, and other team metadata.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_teams.html>
    """
    downloader = get_downloader()

    # Load teams data from nflverse-data repository
    df = downloader.download("nflverse-data", "teams/teams_colors_logos")

    return df
