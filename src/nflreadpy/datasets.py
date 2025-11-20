"""Load nflreadpy datasets."""

import os.path
from importlib import resources

import polars as pl


def data_path(dataset = None) -> str:
    """Get Path to nflreadpy Data Files.

    Returns:
        Path to file. Empty string if `dataset = None` and error if the file doesn't exist.
    """
    if dataset is None:
        return ""
    with resources.path("nflreadpy.data", dataset + ".parquet") as f:
        data_file_path = f
    if os.path.isfile(data_file_path):
        return data_file_path
    else:
        raise FileNotFoundError(f"The file {data_file_path} doesn't exist!")


def team_abbr_mapping() -> pl.DataFrame:
    """Alternate team abbreviation mappings

    A lookup table mapping common alternate team abbreviations.

    Returns:
        Polars DataFrame with two columns `name` and `value` where `value` reflects
        the standardized nflverse team abbreviation.

    See Also:
        <https://nflreadr.nflverse.com/reference/team_abbr_mapping.html>

    """
    return pl.read_parquet(data_path("team_abbr_mapping"))


def team_abbr_mapping_norelocate() -> pl.DataFrame:
    """Alternate team abbreviation mappings, no relocation

    A lookup table mapping common alternate team abbreviations,
    but does not follow relocations to their current city.

    Returns:
        Polars DataFrame with two columns `name` and `value` where `value` reflects
        the standardized nflverse team abbreviation.

    See Also:
        <https://nflreadr.nflverse.com/reference/team_abbr_mapping_norelocate.html>

    """
    return pl.read_parquet(data_path("team_abbr_mapping_norelocate"))
