"""
nflreadpy: A Python package for downloading NFL data from nflverse repositories.

This package provides a Python interface to access NFL data from various
nflverse repositories, with caching, progress tracking, and data validation.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - exercised in packaging workflows
    __version__ = version("nflreadpy")
except PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.0.0"

_EXPORTS = {
    # Core loading functions
    "load_pbp": ".load_pbp",
    "load_player_stats": ".load_stats",
    "load_team_stats": ".load_stats",
    "load_rosters": ".load_rosters",
    "load_schedules": ".load_schedules",
    "load_teams": ".load_teams",
    "load_players": ".load_players",
    "load_draft_picks": ".load_draft_picks",
    "load_injuries": ".load_injuries",
    "load_contracts": ".load_contracts",
    "load_snap_counts": ".load_snap_counts",
    "load_nextgen_stats": ".load_nextgen_stats",
    "load_officials": ".load_officials",
    "load_participation": ".load_participation",
    "load_combine": ".load_combine",
    "load_depth_charts": ".load_depth_charts",
    "load_trades": ".load_trades",
    "load_ftn_charting": ".load_ftn_charting",
    "load_rosters_weekly": ".load_rosters_weekly",
    # ffverse functions
    "load_ff_playerids": ".load_ffverse",
    "load_ff_rankings": ".load_ffverse",
    "load_ff_opportunity": ".load_ffverse",
    # Utility functions
    "get_current_season": ".utils_date",
    "get_current_week": ".utils_date",
    "clear_cache": ".cache",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy importer
    from importlib import import_module

    target_module = _EXPORTS.get(name)
    if not target_module:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = import_module(target_module, __name__)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(list(globals().keys()) + __all__)
