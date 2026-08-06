import warnings

import polars as pl

from .config import get_config
from .datasets import team_abbr_mapping, team_abbr_mapping_norelocate


def clean_team_abbrs(
    abbr: str | list[str], current_location: bool = True, keep_non_matches: bool = True
) -> list:
    """
    Standardize NFL Team Abbreviations.

    Args:
        abbr: a string or list of strings of abbreviations, full team names, or team nicknames.
        current_location: If `True` (the default), the abbreviation of the most recent team
            location will be used.
        keep_non_matches: If `TRUE` (the default) an element of `abbr` that can't
            be matched to any of the internal mappings will be kept as is.
            Otherwise it will be replaced with `None`.

    Returns:
        A string list with the length of `abbr` and cleaned team abbreviations\
        if they are included in `team_abbr_mapping()` or `team_abbr_mapping_norelocate()`\
        (depending on the value of `current_location`). Non matches may be replaced\
        with `Nome` (depending on the value of `keep_non_matches`).
    """
    # if abbr is a single string, we split it to get a list
    if isinstance(abbr, str):
        abbr = abbr.split()

    # error if abbr is no list
    if not isinstance(abbr, list):
        raise TypeError("argument `abbr` must be a string or a list of strings")

    # make sure all elements of the abbr list are string
    if not all(isinstance(elem, str) for elem in abbr):
        raise TypeError("all values of `abbr` must be of type string")

    # make sure other args are boolean
    if not isinstance(current_location, bool):
        raise TypeError("argument `current_location` must be boolean")
    if not isinstance(keep_non_matches, bool):
        raise TypeError("argument `keep_non_matches` must be boolean")

    # load relevant mapping
    mapping = (
        team_abbr_mapping()
        if current_location is True
        else team_abbr_mapping_norelocate()
    )

    # mapping is a polars df. Convert it to a dictionary. We could do the conversion with
    # a polars join but the code below is just easier to read.
    map_dict = dict(mapping.iter_rows())

    # lookup with .get method because it replaces nonmatches with a default value (None)
    out = [map_dict.get(key.upper()) for key in abbr]

    # print list of non matches in a warning if verbose = True
    config = get_config()
    if config.verbose:
        # I couldn't figure out a nice non polars way to compute the list of nonmatches
        df = pl.DataFrame({"old": abbr, "new": out})
        nomatch = df.filter(pl.col("new").is_null()).get_column("old").to_list()
        if len(nomatch) > 0:
            warnings.warn(
                f"Abbreviations not found in `team_abbr_mapping()`: {', '.join(nomatch[:10])}",
                stacklevel=2,
            )

    # out dropped nonmatches. We replace the None values here if the user wants to keep them
    if keep_non_matches is True:
        for index, item in enumerate(out):
            if item is None:
                out[index] = abbr[index]

    return out
