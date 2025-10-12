# Load Functions

nflreadpy provides loader utilities that wrap the downloader and caching layer to
return Polars DataFrames for common nflverse datasets. The table below lists the
most frequently used helpers and the type of data they expose.

| Function | Description | Typical filters | Output schema |
| --- | --- | --- | --- |
| `load_pbp(seasons)` | Play-by-play data for regular and postseason games. | `seasons`, `week`, `team` | One row per play with columns matching the nflverse pbp spec. |
| `load_player_stats(seasons, summary_level)` | Offensive and defensive player stats aggregated by week or season. | `seasons`, `summary_level` | Aggregated statistics keyed by player and season/week. |
| `load_team_stats(seasons, summary_level)` | Team-level stats across rushing, passing, and efficiency splits. | `seasons`, `summary_level` | Team aggregates keyed by team and season/week. |
| `load_schedules(seasons)` | Game schedule, results, and betting lines. | `seasons` | One row per game with kickoff, venue, and line information. |
| `load_rosters(seasons)` | Historical roster data for every franchise. | `seasons` | Player-season membership with position, jersey number, and status. |
| `load_rosters_weekly(seasons)` | Weekly roster snapshots. | `seasons`, `weeks` | Player-week records with roster designations. |
| `load_players()` | Master player index with ids and metadata. | None | Player reference table used for joins. |
| `load_depth_charts(seasons)` | Offense, defense, and special teams depth charts. | `seasons` | Team, position, and depth order per season. |
| `load_snap_counts(seasons)` | Offensive and defensive snap counts. | `seasons` | Player-level snap totals and rates per game. |
| `load_nextgen_stats(seasons)` | Next Gen Stats tracking summaries. | `seasons` | Player and team tracking metrics such as speed and separation. |
| `load_injuries(seasons)` | Practice reports and game status designations. | `seasons` | Injury report entries keyed by team, week, and player. |
| `load_draft_picks()` | NFL draft selections with team and player info. | `seasons` | Draft slot, player, team, and compensation data. |
| `load_trades()` | Historical trade transactions. | `seasons` | Trade details with assets exchanged and teams involved. |
| `load_ff_playerids()` | Crosswalk between nflverse ids and fantasy providers. | None | Mapping table for fantasy integrations. |
| `load_ff_rankings()` | FantasyPros rankings snapshots. | `season`, `week` | Ranked projections with scoring settings. |
| `load_ff_opportunity()` | Expected fantasy opportunity metrics. | `season`, `week` | Player opportunity metrics derived from ffverse data. |

All loader functions accept Python scalars, iterables, or `True` (for "all
available") season arguments. When multiple seasons are requested the resulting
DataFrame is concatenated before being returned. The functions return Polars
DataFrames; convert to pandas with `.to_pandas()` if required for downstream
libraries.
