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
    """
    downloader = get_downloader()

    # Load trades data from nfldata repository (CSV format since RDS isn't readable)
    df = downloader.download("nfldata", "trades", format_preference=DataFormat.CSV)

    return df
