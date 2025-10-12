# Operations Guide

This runbook collects operational practices for the betting toolkit, including
metrics instrumentation, alerting, and incident response.

## Monitoring

Use the supported CLI commands to monitor scraper health and model output.

- `uv run nflreadpy-betting ingest --interval 0 --retries 0` performs a
  one-off scrape and prints the number of stored quotes. Schedule it via cron
  to surface authentication failures or schema changes early.
- `uv run nflreadpy-betting ingest --interval 60 --jitter 5` keeps the
  collector running continuously. The command streams structured logs that can
  be forwarded to your observability platform.
- `uv run nflreadpy-betting simulate --iterations 25000` evaluates stored odds
  without triggering a new scrape, making it suitable for post-ingestion
  validation dashboards.
- `uv run nflreadpy-betting scan --history-limit 500` highlights line movement
  and expected value drift using previously ingested snapshots.

All commands respect storage, alert, and scheduler defaults defined in
`betting.yaml`, so ensure production overrides are committed alongside code.

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

- **Ingestion downtime**: Alert when scheduled `ingest --interval 0` health
  checks fail or when continuous ingest logs stop producing snapshots within the
  expected cadence.
- **Stale data**: Alert when the most recent stored snapshot age exceeds the
  configured polling interval.
- **Model drift**: Monitor calibration metrics exported by your analytics
  pipelines and fire when Brier score or log loss deviates from historical
  baselines.
- **Portfolio limits**: Send a warning when Kelly stakes or exposure reported by
  the dashboard exceed the configured bankroll caps.

## Runbooks

### Ingestion failures

1. Run a one-off ingest to confirm credentials and API availability:

   ```bash
   uv run nflreadpy-betting ingest --interval 0 --retries 0
   ```

   Investigate any failures reported in the logs or alert sinks.
2. Review scraper logs for HTTP errors or upstream schema changes.
3. Temporarily disable the affected sportsbook via configuration overrides if
   the outage persists.
4. Open an issue with the provider and record the incident in the operations log.

### Model regressions

1. Replay historical odds using the backtest command to reproduce the output:

   ```bash
   uv run nflreadpy-betting backtest --limit 250 --iterations 20000
   ```

   Adjust `--limit` to control the number of stored snapshots to evaluate.
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
