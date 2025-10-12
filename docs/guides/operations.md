# Operations Guide

This runbook captures day-to-day operations for the betting stack, including metrics, alerts, and
incident response playbooks.

## Key metrics

| Metric | Description | Target |
| --- | --- | --- |
| `ingestion.poll_latency` | Time between scheduled and actual scraper polls. | < 2x configured interval |
| `ingestion.success_rate` | Percentage of successful scraper runs per window. | > 99% |
| `ingestion.quote_freshness_seconds` | Age of the newest quote per market. | < 15 seconds for live markets |
| `analytics.simulation_latency` | Duration of Monte Carlo runs per batch. | < 5 seconds for 20k iterations |
| `analytics.edge_count` | Number of opportunities above the value threshold. | Investigate sudden drops to zero |
| `alerts.dispatch_latency` | Delay between edge detection and alert emission. | < 10 seconds |
| `dashboard.render_latency` | Time to refresh dashboards or API responses. | < ingestion cadence |

Emit metrics via your observability stack (Prometheus, OpenTelemetry, etc.) from the ingestion
workers, analytics scheduler, and dashboards.

## Alerting policy

- **Ingestion stalled** – fire when `ingestion.quote_freshness_seconds` exceeds 3× the poll
  interval for any sportsbook or market group.
- **Scraper failure burst** – page on-call when `ingestion.success_rate` falls below 95% over five
  consecutive intervals.
- **Analytics backlog** – warn when `analytics.simulation_latency` stays above the target for three
  consecutive runs; throttle Monte Carlo iterations or scale workers accordingly.
- **Dashboard stale** – alert when `dashboard.render_latency` exceeds the ingestion cadence or when
  health pings fail for more than two consecutive checks.
- **Alert delivery failure** – trigger a warning when no alerts are delivered for six hours despite
  detected edges, or when the alert manager raises repeated exceptions.

Route high-severity alerts (ingestion stalled, alert delivery failure) to paging channels and send
other notifications to Slack or email. Incorporate `uv run nflreadpy-betting validate-config`
checks into your release pipeline so configuration regressions surface before production changes
roll out.

## Runbooks

### Ingestion stalled

1. Check the `nflreadpy-betting ingest` logs for rate-limit or credential errors.
2. Validate upstream APIs manually using `uv run nflreadpy-betting ingest --interval 0 --retries 0`
   to perform a single run.
3. Rotate credentials or reduce polling interval if rate limits are triggered.
4. If storage is full, prune historical quotes with the `scan` command and expand capacity.

### Analytics backlog

1. Review Monte Carlo iterations and bankroll simulation options in `config/betting.yaml`.
2. Temporarily lower `analytics.iterations` or disable bankroll simulations via CLI overrides.
3. Scale out analytics workers or provision more CPU.
4. Verify that ingestion is not flooding analytics with duplicated events.

### Dashboard stale

1. Confirm ingestion metrics are healthy; stale dashboards often stem from stale quotes.
2. Restart the Streamlit and FastAPI processes to clear cached state.
3. Review network connectivity between dashboards and the storage backend.
4. If latency persists, enable verbose logging and capture traces for investigation.

### Alert delivery failure

1. Inspect the alert manager logs for API errors or credential failures.
2. Re-run `uv run nflreadpy-betting simulate --enable-alerts` to emit a test alert.
3. Fallback to manual notifications using exported CSVs from the `scan` command.
4. Escalate to platform engineering if third-party alerting providers experience outages.

Document lessons learned after every incident and feed improvements back into configuration,
monitoring, and testing suites.
