# Betting Toolkit CLI

The `nflreadpy-betting` command-line interface bundles operational utilities for
running scrapers, managing analytics jobs, and inspecting stored data. Install
the betting extras before using the CLI.

## Command overview

| Command | Purpose |
| --- | --- |
| `nflreadpy-betting status` | Display ingestion, analytics, and notification health. |
| `nflreadpy-betting validate-config` | Lint a betting configuration file and report schema violations. |
| `nflreadpy-betting ingest` | Run a one-shot scrape for the specified sportsbooks and markets. |
| `nflreadpy-betting replay` | Reprocess historical odds snapshots through registered models. |
| `nflreadpy-betting opportunities` | List current model edges and Kelly stake recommendations. |
| `nflreadpy-betting export` | Write market or opportunity data to CSV/Parquet for external analysis. |

Invoke `--help` on any subcommand for detailed arguments and examples.

## Health and diagnostics

Use the status command to check service connectivity, cache freshness, and
message queue backlogs:

```bash
uv run nflreadpy-betting status --output table
```

`--output json` emits a machine-readable payload ideal for dashboards and alert
pipelines.

## Configuration validation

Before deploying configuration changes, run:

```bash
uv run nflreadpy-betting validate-config --path config/betting.yaml --strict
```

`--strict` fails on warnings and ensures orchestrated rollouts do not proceed
with suspicious polling intervals, stale credentials, or unknown sportsbooks.

## Ingestion utilities

Kick off an ad-hoc scrape for a subset of sportsbooks and markets:

```bash
uv run nflreadpy-betting ingest --sportsbooks pinnacle,draftkings --markets moneyline,spread --scope live
```

The command stores raw odds, normalised snapshots, and audit logs using the
same storage backend as the always-on ingestion service.

Replay historical data when testing new models or analytics pipelines:

```bash
uv run nflreadpy-betting replay --from 2024-01-01 --to 2024-01-31 --models elo,simulator
```

## Opportunity inspection

Surface active model edges and Kelly stakes without launching the dashboards:

```bash
uv run nflreadpy-betting opportunities --min-edge 1.5 --sport nfl
```

Add `--explain` to include model diagnostics and feature contributions when the
underlying models expose explanation metadata.

## Data export

Bulk export curated datasets for archival or external analysis:

```bash
uv run nflreadpy-betting export --dataset opportunities --format parquet --output data/opportunities.parquet
```

Exports respect configured retention policies, allowing compliance or finance
teams to sample the same snapshots used in daily decision-making.
