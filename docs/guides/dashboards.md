# Dashboard Operations

nflreadpy ships with two complementary dashboards: a terminal-first ASCII layout powered by the
`Dashboard` class and a Streamlit UI backed by the same data provider contracts. Follow the steps
below to run, customise, and troubleshoot each experience.

## Terminal dashboard quick start

```bash
uv run nflreadpy-betting dashboard --refresh --iterations 15000 --value-threshold 0.015
```

The dashboard launches with fresh odds, Monte Carlo simulations, and Kelly sizing summaries. Use
keyboard shortcuts documented in [`docs/betting/dashboard.md`](../betting/dashboard.md) to toggle
panels, apply filters, and trigger manual refreshes.

Common configuration tweaks:

- `--bankroll` and `--portfolio-fraction` control bankroll context and position caps.
- `--correlation-limit group=value` constrains exposure for correlated outcomes (e.g. same game).
- `--risk-trials` enables bankroll drawdown simulations for scenario analysis.

## Streamlit dashboard

Serve the Streamlit view with:

```bash
uv run streamlit run -m nflreadpy.betting.web.app
```

The application surfaces:

- sortable odds tables with shared filters from the terminal dashboard;
- edge summaries with expected value, stake, and Kelly output;
- Vega-Lite charts for line movement and calibration diagnostics; and
- bankroll panels that mirror the CLI risk simulations.

Deploy behind your preferred reverse proxy and set the `DASHBOARD_REFRESH_SECONDS` environment
variable to match the cadence of your ingestion workers.

## FastAPI analytics API

For programmatic access, construct the FastAPI app exposed via
`nflreadpy.betting.web.api:create_api_app`. The service mirrors the dashboard provider contract and
exposes `/markets`, `/opportunities`, `/line-history`, `/portfolio`, and `/filters` endpoints. Use
the `/markets/stream` endpoint to stream newline-delimited JSON snapshots into downstream queues.

```bash
uv run uvicorn nflreadpy.betting.web.api:create_api_app --factory
```

## Observability checklist

- Instrument dashboard processes with the metrics documented in the [operations guide](operations.md).
- Forward Streamlit and FastAPI logs to your logging stack; failed refreshes often surface as 5xxs or
  repeated retries.
- Alert when dashboard latency exceeds ingestion cadence—stale visuals indicate upstream slowness or
  provider outages.
