# nflreadpy v0.1.2
Release date: 2025-09-15

First version of nflreadpy, a port of nflreadr and a successor to nfl_data_py,
with the goal of starting fresh and maintaining API compatibility with nflreadr
so that it's easier for nflverse maintainers to keep it in parallel with nflreadr
updates. This first version was mostly written with the help of Claude Code.

## New functions
The following functions are included in this release:

- load_pbp() - play-by-play data
- load_player_stats() - player game or season statistics
- load_team_stats() - team game or season statistics
- load_schedules() - game schedules and results
- load_players() - player information
- load_rosters() - team rosters
- load_rosters_weekly() - team rosters by season-week
- load_snap_counts() - snap counts
- load_nextgen_stats() - advanced stats from nextgenstats.nfl.com
- load_ftn_charting() - charted stats from ftnfantasy.com/data
- load_participation() - participation data (historical)
- load_draft_picks() - nfl draft picks
- load_injuries() - injury statuses and practice participation
- load_contracts() - historical contract data from OTC
- load_officials() - officials for each game
- load_combine() - nfl combine results
- load_depth_charts() - depth charts
- load_trades() - trades
- load_ff_playerids() - ffverse/dynastyprocess player ids
- load_ff_rankings() - fantasypros rankings
- load_ff_opportunity() - expected yards, touchdowns, and fantasy points
- clear_cache() - Clear cached data
- get_current_season() - Get current NFL season
- get_current_week() - Get current NFL week

## Feature comparisons

- [feature comparison with nflreadr](https://github.com/nflverse/nflreadpy/issues/2)
- [feature comparison with nfl_data_py](https://github.com/nflverse/nflreadpy/issues/6)

## Acknowledgements

Thanks to @mrcaseb, @guga31bb, @guidopetri, and @akeaswaran for reviewing the
code in this release, and to @alecglen and @cooperdff for their stewardship of
the nfl_data_py package.
