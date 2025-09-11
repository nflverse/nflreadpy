"""Load NFL contract data."""

import polars as pl

from .downloader import get_downloader


def load_contracts() -> pl.DataFrame:
    """
    Load NFL historical contract data.

    Returns:
        Polars DataFrame with historical contract information including
        player details, contract terms, values, and team information.

    See Also:
        https://nflreadr.nflverse.com/reference/load_contracts.html
    """
    downloader = get_downloader()

    # Load historical contracts data from nflverse-data repository
    df = downloader.download("nflverse-data", "contracts/historical_contracts")

    return df
