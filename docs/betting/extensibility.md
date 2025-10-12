# Developer Extension Points

The betting toolkit is designed to accommodate custom data sources, models, and
analytics pipelines. This document outlines the primary integration surfaces and
best practices for extending the system.

## Scrapers

Scrapers implement the `OddsScraper` protocol and register themselves with the
ingestion registry.

```python
from nflreadpy.betting.ingestion import OddsScraper, register_scraper

class MySportsbookScraper(OddsScraper):
    name = "mysportsbook"

    async def fetch_markets(self, *, markets, scope, regions):
        response = await self._client.get(...)
        return self._normalise(response.json())

register_scraper(MySportsbookScraper)
```

### Guidelines

- **Normalisation**: Return market quotes using canonical field names (`event_id`,
  `market`, `selection`, `price`, `limit`, `updated_at`).
- **Resilience**: Catch provider-specific errors and raise `ScraperTransientError`
  to trigger retries without poisoning the ingestion pipeline.
- **Telemetry**: Emit metrics via `nflreadpy.betting.telemetry` for latency,
  error rates, and throttling occurrences.

Register new scrapers by referencing their module path inside the betting
configuration file:

```yaml
ingestion:
  sportsbooks:
    - name: mysportsbook
      loader: myproject.scrapers.mysportsbook:MySportsbookScraper
```

## Models

Models adhere to the `BettingModel` protocol and produce probabilities,
expected values, and stakes for each market snapshot.

```python
from nflreadpy.betting.models import BettingModel, ModelContext

class MyModel(BettingModel):
    name = "my_model"

    def fit(self, training_data: ModelContext) -> None:
        self.pipeline = ...

    def predict(self, market: ModelContext) -> dict:
        return {
            "prob": self.pipeline.predict_proba(market.features),
            "edge": ...,
            "kelly": ...,
        }
```

Register the model by exposing it through an entry point or listing it inside
`analytics.models` in the betting configuration file.

### Best practices

- **Feature stores**: Use the provided `FeatureService` abstractions to retrieve
  consistent features across offline training and online inference.
- **Versioning**: Set the `model_version` attribute and log provenance metadata
  when emitting opportunities.
- **Backtesting**: Use `nflreadpy-betting backtest` to validate new models
  against historical snapshots before promoting them to production.

## Analytics pipelines

Analytics modules extend `AnalyticsTask` and are orchestrated by the scheduler.

```python
from nflreadpy.betting.analytics import AnalyticsTask

class ExposureLimitsTask(AnalyticsTask):
    name = "exposure_limits"

    async def run(self, context):
        portfolio = await context.get_portfolio()
        alerts = compute_limits(portfolio)
        await context.emit_alerts(alerts)
```

Configure recurring schedules in `analytics.tasks` with cron-style expressions or
intervals.

```yaml
analytics:
  tasks:
    - module: myproject.analytics.exposure:ExposureLimitsTask
      schedule: "*/5 * * * *"
```

## Testing extensions

- Write unit tests that mock external providers and assert the normalised data
  structure.
- Use `pytest` fixtures to stub alert sinks or storage backends when verifying
  side effects.
- Run `uv run nflreadpy-betting backtest` against curated fixtures to compare model
  output before and after changes.

## Distribution

Package scrapers, models, and analytics tasks in a dedicated Python distribution
and declare optional dependencies for sportsbook-specific SDKs. Consumers can
install the package alongside `nflreadpy[betting]` to enable your integrations.
