# User Guide

This guide walks analysts and trading desk operators through the standard
nflreadpy betting workflow. It focuses on running the stack with the
bundled defaults so you can ingest odds, surface edges, and monitor
opportunities without touching source code.

## Setup

1. Install the betting extras:
   ```bash
   uv sync --all-extras --dev
   ```
   or use the published Docker images:
   ```bash
   docker run --rm ghcr.io/nflverse/nflreadpy:latest-cpu --help
   ```
2. Copy the sample configuration into your working directory:
   ```bash
   cp config/betting.yaml my-config.yaml
   ```
3. Export a writable storage directory (defaults to
   `data/betting_odds.sqlite3`).

For container users, mount a host directory so SQLite snapshots persist
between runs:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  ghcr.io/nflverse/nflreadpy:latest-cpu ingest --interval 0
```

## Configuration

Edit `my-config.yaml` to enable the scrapers you need. Each scraper block
controls parameters (endpoints, authentication headers) and runtime
behaviour (poll cadence, retry limits, timeouts). Environment-specific
files such as `config/betting.production.yaml` overlay the base file
when you set `NFLREADPY_BETTING_ENV`. You can make one-off tweaks with
environment overrides:
```bash
export NFLREADPY_BETTING__analytics__value_threshold=0.025
```

Validate your configuration before deploying:
```bash
uv run nflreadpy-betting validate-config --config my-config.yaml
```

## CLI

The CLI exposes one-shot and long-running workflows:

- `uv run nflreadpy-betting ingest --interval 0` performs a single odds
  refresh using the scrapers defined in your configuration.
- `uv run nflreadpy-betting scan --iterations 12000` ranks opportunities
  above the configured value threshold.
- `uv run nflreadpy-betting simulate --refresh` combines fresh odds,
  Monte Carlo simulations, and bankroll sizing heuristics.

Use `--storage` to point at an alternate SQLite file and
`--alerts-config` to supply alert routing secrets (see alerts section).

## Dashboards

Launch the terminal dashboard for a Bloomberg-style snapshot:
```bash
uv run nflreadpy-betting dashboard --refresh --iterations 15000
```

For a web experience run the Streamlit app:
```bash
uv run streamlit run -m nflreadpy.betting.web.app
```

Both dashboards honour the configuration defaults for bankroll, value
thresholds, and correlation limits. Use the keyboard shortcuts and query
parameters described in `docs/guides/dashboards.md` to tailor the view to
your markets.

## Metrics

The ingestion service and dashboard emit structured metrics via the
`metrics` property and the audit logger. At a minimum track:

- `ingestion.requested` and `ingestion.persisted` to verify healthy odds
  flow.
- `ingestion.latency_seconds` for scrape performance.
- `analytics.edge_count` and `analytics.simulation_latency` to make sure
  modeling keeps pace with ingestion.

Feed those counters into Prometheus or your preferred observability
stack to trigger alert thresholds.

## Alerts

Define alert routing in a YAML file consumed by `AlertManager` (see
`docs/betting/operations.md`). Pass the path with
`--alerts-config /path/to/alerts.yaml` so the CLI can emit Slack, email,
or SMS notifications when an edge clears the configured value
threshold. Test the wiring with:
```bash
uv run nflreadpy-betting simulate --refresh --value-threshold 0.05 --alerts-config alerts.yaml
```

Alerts are rate-limited and deduplicated automatically; the CLI prints a
summary of dispatched notifications for quick verification.
