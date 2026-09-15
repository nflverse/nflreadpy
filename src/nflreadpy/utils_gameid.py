"""Game ID utility for nflreadpy."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from .datasets import team_abbr_mapping_norelocate
from .utils_date import get_current_season

_MIN_SEASON = 1999
_MAX_WEEK = 22


def _as_list(value: object) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _team_abbr_lookup() -> dict[str, str]:
    """Build a name -> standardized abbreviation lookup, first match wins."""
    lookup: dict[str, str] = {}
    for row in team_abbr_mapping_norelocate().iter_rows(named=True):
        lookup.setdefault(row["name"], row["value"])
    return lookup


def nflverse_game_id(
    season: int | Sequence[int],
    week: int | Sequence[int],
    away: str | Sequence[str],
    home: str | Sequence[str],
) -> str | list[str]:
    """
    Compute nflverse Game Identifiers.

    Args:
        season: 4 digit season(s) between 1999 and the current season.
        week: Week number(s) between 1 and 22.
        away: Away team abbreviation(s), full name(s), or nickname(s), as found
            in `team_abbr_mapping_norelocate`.
        home: Home team abbreviation(s), full name(s), or nickname(s), as found
            in `team_abbr_mapping_norelocate`.

    Returns:
        A single game ID string if all arguments are scalars, otherwise a list
        of game ID strings (arguments are recycled to the longest argument's
        length, matching R's `nflverse_game_id()` behavior).

    See Also:
        <https://nflreadr.nflverse.com/reference/nflverse_game_id.html>
    """
    scalar_input = not any(
        isinstance(v, (list, tuple)) for v in (season, week, away, home)
    )

    seasons = _as_list(season)
    weeks = _as_list(week)
    aways = _as_list(away)
    homes = _as_list(home)

    max_season = get_current_season(roster=True)
    invalid_seasons = sorted(
        {s for s in seasons if not (_MIN_SEASON <= int(s) <= max_season)}
    )
    if invalid_seasons:
        raise ValueError(
            f"`season` must be between {_MIN_SEASON} and {max_season}, "
            f"got {invalid_seasons}"
        )

    invalid_weeks = sorted({w for w in weeks if not (1 <= int(w) <= _MAX_WEEK)})
    if invalid_weeks:
        raise ValueError(
            f"`week` must be between 1 and {_MAX_WEEK}, got {invalid_weeks}"
        )

    lookup = _team_abbr_lookup()

    invalid_home = sorted({h for h in homes if h.upper() not in lookup})
    if invalid_home:
        raise ValueError(f"Invalid `home` abbreviation(s): {invalid_home}")

    invalid_away = sorted({a for a in aways if a.upper() not in lookup})
    if invalid_away:
        raise ValueError(f"Invalid `away` abbreviation(s): {invalid_away}")

    lengths = {len(seasons), len(weeks), len(aways), len(homes)}
    max_length = max(lengths)
    if not lengths <= {1, max_length}:
        warnings.warn(
            "Arguments should be of length one or of length "
            f"{max_length}. Recycling may produce unexpected results.",
            stacklevel=2,
        )

    def recycle(values: list, length: int) -> list:
        return [values[i % len(values)] for i in range(length)]

    recycled_seasons = recycle([int(s) for s in seasons], max_length)
    recycled_weeks = recycle([int(w) for w in weeks], max_length)
    recycled_aways = recycle([lookup[a.upper()] for a in aways], max_length)
    recycled_homes = recycle([lookup[h.upper()] for h in homes], max_length)

    game_ids = [
        f"{s}_{w:02d}_{a}_{h}"
        for s, w, a, h in zip(
            recycled_seasons,
            recycled_weeks,
            recycled_aways,
            recycled_homes,
            strict=True,
        )
    ]

    return game_ids[0] if scalar_input else game_ids
