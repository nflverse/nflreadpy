# Platform Setup

Follow these steps to prepare a repeatable environment for the nflreadpy betting stack.

## Create an isolated environment

Install the project with the betting extras to pull in the ingestion workers, dashboards, and
CLI entry points:

```bash
uv sync --locked --group betting
```

The `betting` extra installs Streamlit, FastAPI, and Uvicorn alongside the base package. Use
`uv venv` or your preferred virtual environment manager to keep dependencies isolated per
deployment.

## Configure environment variables

The runtime reads most secrets from environment variables. Export the variables directly or
store them in a `.env` file consumed by your process manager.

```bash
export NFLREADPY_CACHE_MODE="filesystem"
export NFLREADPY_CACHE_DIR="$HOME/.cache/nflreadpy"
export SPORTSBOOK_API_KEY="..."
export SPORTSBOOK_API_SECRET="..."
```

Set `NFLREADPY_VERBOSE=true` when troubleshooting HTTP downloads, and provide a
`NFLREADPY_USER_AGENT` that identifies your integration if sportsbooks require it.

## Prime reference data

Run the primer script once after installation (or whenever nflverse publishes a new season) to
warm the cache with schedules, rosters, and depth charts:

```python
import nflreadpy as nfl

for loader in (nfl.load_schedules, nfl.load_rosters, nfl.load_depth_charts):
    loader(seasons=True)
```

Caching the shared datasets prevents the ingestion and analytics schedulers from blocking on
first access.

## Start the core services

Launch the ingestion and analytics pipelines in separate terminals or via a supervisor. Each
service exposes structured logging you can forward to your observability stack.

```bash
uv run python -m nflreadpy.betting.ingestion.service
uv run python -m nflreadpy.betting.analytics.scheduler
```

Follow with `uv run nflreadpy-betting status` to confirm scraper connectivity, model health,
and alert delivery before moving to production.
