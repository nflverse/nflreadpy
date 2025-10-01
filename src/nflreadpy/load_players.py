"""Load NFL player data."""

import polars as pl

from .downloader import get_downloader


def load_players() -> pl.DataFrame:
    """
    Load NFL player information.

    This is a comprehensive source of player information including basic details,
    draft information, positions, and ID mappings across multiple data sources
    (GSIS, PFR, PFF, OTC, ESB, ESPN).

    Returns:
        Polars DataFrame with player data - one row per player with comprehensive \
        player information including names, physical stats, draft info, and \
        cross-platform ID mappings.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_players.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_players.html>
    """
    downloader = get_downloader()

    # Load players data from nflverse-data repository
    df = downloader.download("nflverse-data", "players/players")

    return df
