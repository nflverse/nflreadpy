# Betting Toolkit Setup

This guide walks through installing the betting extras, preparing market data
feeds, and verifying the toolkit is ready for day-to-day use.

## 1. Install dependencies

The betting module ships as an optional extra. Install the core package along
with the optional web stack for the Streamlit dashboard and FastAPI service:

```bash
uv add "nflreadpy[betting]"
```

If you prefer to pin dependencies inside a project-managed environment, sync
the `pyproject.toml` lock file instead:

```bash
uv sync --group betting
```

!!! tip
    The command installs Streamlit, FastAPI, and Uvicorn in addition to the core
    nflreadpy requirements. Skip the `[betting]` extra when you only need the
    terminal dashboard or CLI utilities.

## 2. Configure credentials

Some sportsbooks and odds providers rate-limit unauthenticated traffic. Export
API keys and credentials before launching the ingestion services:

```bash
export SPORTSBOOK_API_KEY="..."
export SPORTSBOOK_API_SECRET="..."
```

The ingest layer reads from environment variables by default. Alternatively,
create a `.env` file at the project root and load it with `python-dotenv` or
your preferred process manager.

## 3. Provision cache storage

Live odds ingestion can produce sizable snapshots. Enable filesystem caching to
persist raw pulls and model output across restarts:

```bash
uv run python -c "import nflreadpy as nfl; nfl.update_config(cache_mode='filesystem', cache_path='.cache/nflreadpy', cache_duration=300)"
```

Shorter cache durations (3–5 minutes) keep the dashboards responsive while
preventing stale prices from propagating.

## 4. Seed reference data

Initial simulations rely on league metadata such as team mappings and depth
charts. Pre-download the canonical datasets so downstream jobs do not block on
first use:

```python
import nflreadpy as nfl

nfl.load_schedules(seasons=True)
nfl.load_rosters(seasons=True)
nfl.load_depth_charts(seasons=True)
```

Run the snippet once after installing or when nflverse publishes a new season.
The cache layer stores the data for subsequent sessions.

## 5. Start background services

The toolkit expects a continual stream of market quotes and model projections.
Launch the ingestion worker alongside the analytics scheduler:

```bash
uv run python -m nflreadpy.betting.ingestion.service
uv run python -m nflreadpy.betting.analytics.scheduler
```

Each process emits structured logs describing scraper health, latency, and
throughput. Surface the log directories in your observability stack to spot
upstream outages early.

## 6. Verify the installation

With ingestion running, execute the health check command:

```bash
uv run nflreadpy-betting status
```

The output confirms connectivity to sportsbooks, model registries, and the
analytics database. Investigate any reported failures before deploying to
production environments.
