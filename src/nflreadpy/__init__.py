"""
nflreadpy: A Python package for downloading NFL data from nflverse repositories.

This package provides a Python interface to access NFL data from various
nflverse repositories, with caching, progress tracking, and data validation.
"""

__version__ = "0.1.0"

from .cache import clear_cache
from .load_combine import load_combine
from .load_contracts import load_contracts
from .load_depth_charts import load_depth_charts
from .load_draft_picks import load_draft_picks
from .load_ffverse import load_ff_opportunity, load_ff_playerids, load_ff_rankings
from .load_ftn_charting import load_ftn_charting
from .load_injuries import load_injuries
from .load_nextgen_stats import load_nextgen_stats
from .load_officials import load_officials
from .load_participation import load_participation
from .load_pbp import load_pbp
from .load_players import load_players
from .load_rosters import load_rosters
from .load_rosters_weekly import load_rosters_weekly
from .load_schedules import load_schedules
from .load_snap_counts import load_snap_counts
from .load_stats import load_player_stats, load_team_stats
from .load_teams import load_teams
from .load_trades import load_trades
from .utils_date import get_current_season, get_current_week

__all__ = [
    # Core loading functions
    "load_pbp",
    "load_player_stats",
    "load_team_stats",
    "load_rosters",
    "load_schedules",
    "load_teams",
    "load_players",
    "load_draft_picks",
    "load_injuries",
    "load_contracts",
    "load_snap_counts",
    "load_nextgen_stats",
    "load_officials",
    "load_participation",
    "load_combine",
    "load_depth_charts",
    "load_trades",
    "load_ftn_charting",
    "load_rosters_weekly",
    # ffverse functions
    "load_ff_playerids",
    "load_ff_rankings",
    "load_ff_opportunity",
    # Utility functions
    "get_current_season",
    "get_current_week",
    "clear_cache",
]
