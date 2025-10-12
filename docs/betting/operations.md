# Operations Guide

This runbook collects operational practices for the betting toolkit, including
metrics instrumentation, alerting, and incident response.

## Metrics

The betting module emits Prometheus-formatted metrics through the telemetry
package. Expose the `/metrics` endpoint by running the telemetry server:

```bash
uv run python -m nflreadpy.betting.telemetry --bind 0.0.0.0 --port 9100
```

Key metrics include:

- `nflreadpy_ingestion_latency_seconds` (histogram): time to fetch and normalise
a snapshot per sportsbook and market.
- `nflreadpy_ingestion_errors_total` (counter): count of scraper failures by
error type.
- `nflreadpy_model_edge` (gauge): most recent expected value and Kelly fraction
per market.
- `nflreadpy_portfolio_exposure` (gauge): current exposure by team, market, and
book.

Scrapers and analytics tasks automatically register metrics collectors. Extend
the telemetry configuration in `betting.yaml` to set custom namespaces, push
intervals, or remote write targets.

## Alert configuration

Route alerts to your collaboration tools using the notifications block in the
betting configuration file. Common examples:

```yaml
notifications:
  slack:
    webhook_url: https://hooks.slack.com/services/.../...
    channel: "#betting-alerts"
    severity_threshold: warning
  pagerduty:
    routing_key: PAGERDUTY_KEY
    severity_threshold: critical
```

### Recommended alerts

- **Ingestion downtime**: Trigger when `nflreadpy_ingestion_errors_total` grows
  faster than expected or when status checks fail.
- **Stale data**: Alert when the most recent snapshot age exceeds the configured
  polling interval.
- **Model drift**: Monitor calibration metrics and fire when Brier score or log
  loss deviates from historical baselines.
- **Portfolio limits**: Send a warning when Kelly stakes or exposure exceed the
  configured bankroll caps.

## Runbooks

### Ingestion failures

1. Check the status command for failing sportsbooks:

   ```bash
   uv run nflreadpy-betting status --output table
   ```

2. Review scraper logs for HTTP errors or upstream schema changes.
3. Temporarily disable the affected sportsbook via configuration overrides if
   the outage persists.
4. Open an issue with the provider and record the incident in the operations log.

### Model regressions

1. Replay the failing interval to reproduce the output:

   ```bash
   uv run nflreadpy-betting replay --from <start> --to <end> --models <model>
   ```

2. Compare calibration metrics with the `analytics.metrics` dashboard panels.
3. Roll back to the previous model version if confidence intervals degrade
   significantly.
4. Schedule an RCA documenting feature drift, data quality issues, or code
   regressions.

### Dashboard outages

1. Confirm ingestion and analytics services are healthy.
2. Restart the Streamlit or FastAPI process and check application logs.
3. Inspect CDN or reverse proxy configuration for TLS or routing issues.
4. Notify users via Slack or email and provide an estimated time to resolution.

## Change management

- Use feature flags to roll out new scrapers or models gradually.
- Record configuration changes in version control or an auditable change log.
- Pair configuration deployments with `validate-config --strict` to catch
  invalid state before it reaches production.

## Disaster recovery

- Schedule regular backups of the analytics database and object storage buckets
  holding raw snapshots.
- Test restore procedures quarterly using a staging environment.
- Maintain warm standby infrastructure for critical services during peak events
  (e.g., playoffs, Super Bowl).
