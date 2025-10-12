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

## Run with the published container images

CI builds multi-arch container images for the betting stack and publishes them to GitHub
Container Registry. Use the CPU-optimised image for general deployments and the CUDA variant when
running GPU-accelerated simulations.

```bash
# CPU image
docker run --rm \
  -e NFLREADPY_BETTING_ENV=production \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  ghcr.io/nflverse/nflreadpy:latest-cpu ingest --interval 60 --jitter 5

# GPU image
docker run --rm --gpus all \
  -e NFLREADPY_BETTING_ENV=production \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  ghcr.io/nflverse/nflreadpy:latest-gpu simulate --iterations 40000 --refresh
```

Both images ship with the betting extras installed and use `nflreadpy-betting` as the default
entrypoint, so invoking them with no additional arguments prints CLI help. Mount your configuration
and state directories into `/app/config` and `/app/data` to persist overrides and SQLite storage.

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

Launch the ingestion and analytics pipelines in separate terminals or via a supervisor using the
supported CLI commands. Each command emits structured logging you can forward to your observability
stack.

```bash
uv run nflreadpy-betting ingest --interval 60 --jitter 5
uv run nflreadpy-betting simulate --refresh --iterations 25000
```

Follow with a one-off scrape to verify connectivity, storage permissions, and alert delivery before
moving to production:

```bash
uv run nflreadpy-betting ingest --interval 0 --retries 0
```
