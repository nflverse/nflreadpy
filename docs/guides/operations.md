# Operations Guide

This runbook covers everything operations teams need to run the betting
stack in production—from environment bootstrap through monitoring and
incident response. Each section maps to the lifecycle topics surfaced in
README.md so you can jump straight to the task at hand.

## Setup

- Provision compute capable of running the CPU or GPU Docker images.
- Mount persistent storage for the ingestion SQLite database and logs.
- Configure secrets management for sportsbook credentials and alerting
tokens.
- Schedule configuration validation during deployments:
  ```bash
  uv run nflreadpy-betting validate-config --config config/betting.yaml
  ```

## Configuration

Layered configuration ensures small tweaks do not require editing the
base file:

1. `config/betting.yaml` captures shared defaults.
2. `config/betting.<env>.yaml` overrides environment specifics when
   `NFLREADPY_BETTING_ENV` is set (e.g. `production`).
3. `NFLREADPY_BETTING_CONFIG` points to additional YAML overrides (colon
   separated list).
4. Environment overrides such as
   `NFLREADPY_BETTING__ingestion__storage_path=/var/lib/nflreadpy/odds.sqlite3`
   are merged last.

Before each rollout run `validate-config` with the target environment and
ensure credentials referenced via `${VAR}` tokens exist.

## CLI

Operations teams primarily use three commands:

- `ingest` to run the scheduler continuously or in single-shot mode.
- `simulate` to confirm Monte Carlo health against stored odds.
- `scan` to export candidate opportunities for manual review.

Provide `--alerts-config` to route notifications, and pass `--storage`
when restoring from backups or rotating storage hosts. The CLI prints
metrics snapshots and warnings that should be forwarded to your logging
platform.

## Dashboards

- Terminal dashboard: `uv run nflreadpy-betting dashboard --refresh`. Use
  it during incidents to verify odds freshness and model output without a
  browser.
- Streamlit dashboard: `uv run streamlit run -m nflreadpy.betting.web.app`
  for the trading desk UI. Configure process managers (systemd, ECS,
  Kubernetes) to restart the app on failure and to scrape health probes.
- FastAPI service: `uv run uvicorn nflreadpy.betting.web.api:create_api_app --factory`
  to expose REST endpoints for downstream automation.

Ensure dashboards and APIs run close to the ingestion database to avoid
stale reads caused by network jitter.

## Metrics

Track the following metrics to understand system health:

| Metric | Description | Target |
| --- | --- | --- |
| `ingestion.poll_latency` | Time between scheduled and actual scraper polls. | < 2× configured interval |
| `ingestion.success_rate` | Percentage of successful scraper runs per window. | > 99% |
| `ingestion.quote_freshness_seconds` | Age of the newest quote per market. | < 15 seconds for live markets |
| `analytics.simulation_latency` | Duration of Monte Carlo runs per batch. | < 5 seconds for 20k iterations |
| `analytics.edge_count` | Number of opportunities above the value threshold. | Investigate sudden drops to zero |
| `alerts.dispatch_latency` | Delay between edge detection and alert emission. | < 10 seconds |
| `dashboard.render_latency` | Time to refresh dashboards or API responses. | < ingestion cadence |

Emit metrics via your observability stack (Prometheus, OpenTelemetry,
Datadog) from the ingestion workers, analytics scheduler, dashboards, and
alerting adapters.

## Alerts

Recommended alerting policy:

- **Ingestion stalled** – fire when `ingestion.quote_freshness_seconds`
  exceeds 3× the poll interval for any sportsbook or market group.
- **Scraper failure burst** – page on-call when
  `ingestion.success_rate` drops below 95% over five consecutive
  intervals.
- **Analytics backlog** – warn when `analytics.simulation_latency`
  breaches target for three runs; scale workers or reduce iterations.
- **Dashboard stale** – alert when `dashboard.render_latency` exceeds the
  ingestion cadence or health checks fail repeatedly.
- **Alert delivery failure** – trigger a warning when no alerts are
  delivered for six hours despite detected edges, or when the alert
  manager raises repeated exceptions.

Route high-severity alerts to paging channels and send informational
notifications to Slack or email. Test the pipeline with
`nflreadpy-betting simulate --refresh --alerts-config alerts.yaml`.

## Runbooks

### Ingestion stalled

1. Check `nflreadpy-betting ingest` logs for rate-limit or credential
   errors.
2. Validate upstream APIs manually using
   `uv run nflreadpy-betting ingest --interval 0 --retries 0`.
3. Rotate credentials or reduce polling interval if rate limits are
   triggered.
4. If storage is full, prune historical quotes with the `scan` command
   and expand capacity.

### Analytics backlog

1. Review Monte Carlo iterations and bankroll simulation options in
   `config/betting.yaml`.
2. Temporarily lower `analytics.iterations` or disable bankroll
   simulations via CLI overrides.
3. Scale out analytics workers or provision more CPU/GPU resources.
4. Verify ingestion is not flooding analytics with duplicated events.

### Dashboard stale

1. Confirm ingestion metrics are healthy; stale dashboards often stem
   from stale quotes.
2. Restart the Streamlit and FastAPI processes to clear cached state.
3. Review network connectivity between dashboards and the storage
   backend.
4. Enable verbose logging and capture traces if latency persists.

### Alert delivery failure

1. Inspect alert manager logs for API errors or credential failures.
2. Re-run `uv run nflreadpy-betting simulate --enable-alerts` to emit a
   test alert.
3. Fallback to manual notifications using exported CSVs from the `scan`
   command.
4. Escalate to platform engineering if third-party alerting providers
   experience outages.

Document lessons learned after every incident and feed improvements back
into configuration, monitoring, and testing suites.
