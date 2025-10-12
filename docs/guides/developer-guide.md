# Developer Guide

This guide explains how to extend the betting stack, customise
configuration layers, and integrate the tooling into existing research or
trading automation.

## Setup

Clone the repository and create a development environment with uv:
```bash
uv sync --all-extras --dev
```

Enable pre-commit hooks and mypy checks if you plan to contribute code:
```bash
uv run pre-commit install
uv run pre-commit run --all-files
uv run mypy src
```

Docker images (`Dockerfile.cpu` and `Dockerfile.gpu`) mirror the
recommended runtime stack; build them locally to exercise your changes in
an isolated environment:
```bash
docker build -f Dockerfile.cpu -t nflreadpy-betting:dev .
```

## Configuration

Configuration is modelled with Pydantic and loaded through
`load_betting_config`. The loader applies layers in the following order:

1. `config/betting.yaml` (or the path passed via `--config`).
2. Environment-specific overrides (e.g. `config/betting.production.yaml`)
   when `NFLREADPY_BETTING_ENV` is set.
3. Extra override files supplied via `NFLREADPY_BETTING_CONFIG` (colon
   separated).
4. Deep overrides from `NFLREADPY_BETTING__*` environment variables.
5. `${VAR}` tokens interpolated from the environment.

Use `BettingConfig.model_validate` to assert schema correctness in unit
tests and `validate_betting_config` before constructing services. Export
helper factories (`create_scrapers_from_config`,
`create_ingestion_service`) when wiring custom workflows.

## CLI

The CLI in `nflreadpy.betting.cli` is built around the `SubcommandApp`
registry. Add new subcommands by decorating async handlers with
`@APP.command`. Reuse `_apply_config_defaults` so CLI arguments align
with configuration defaults. When adding commands, include integration
tests under `tests/betting/` that exercise the new flags using `uv run`
subprocess calls, mirroring the existing `validate-config` test.

## Dashboards

Two dashboard implementations ship with the project:

- `Dashboard` (curses/terminal) in `dashboard_tui.py`.
- Streamlit and FastAPI apps under `betting/web/`.

Extend dashboards by adding new panels to `dashboard_core.py` and
`dashboard_tui.py`. Keep the provider contracts stable so the web and
terminal experiences stay in sync. For bespoke front-ends, import the
`DashboardSnapshot` dataclass and render it inside your preferred UI
framework.

## Metrics

The ingestion service exposes counters via the `metrics` property and
pushes structured events to the audit logger. When adding new workflows
or background jobs, update the metrics dictionary so operations teams can
observe throughput, latency, and error rates. The tests under
`tests/betting/test_ingestion.py` demonstrate how to assert against
metrics snapshots without touching a live database.

Instrument additional code paths with your observability framework of
choice (Prometheus client, OpenTelemetry). Avoid expensive calls inside
critical ingestion loops; instead collect metrics after each poll cycle
and emit them asynchronously if needed.

## Alerts

Alert delivery is orchestrated by `AlertManager` and sink implementations
in `alerts.py`. When integrating a new alert channel, subclass
`AlertSink`, implement the `send` coroutine, and register it inside
`get_alert_manager`. Alert routing is defined in YAML files and passed to
the CLI via `--alerts-config`. Use the developer sandbox to emit dry-run
alerts before enabling production channels.
