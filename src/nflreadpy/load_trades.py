"""Load NFL trades data."""

import polars as pl

from .config import DataFormat
from .downloader import get_downloader


def load_trades() -> pl.DataFrame:
    """
    Load NFL trades data.

    Returns:
        Polars DataFrame with NFL trade information including players,
        teams, draft picks, and trade details.

    See Also:
        <https://nflreadr.nflverse.com/reference/load_trades.html>

    Data Dictionary:
        <https://nflreadr.nflverse.com/articles/dictionary_trades.html>
    """
    downloader = get_downloader()

    # Load trades data from nfldata repository (CSV format since RDS isn't readable)
    df = downloader.download("nfldata", "trades", format=DataFormat.CSV)

    return df
