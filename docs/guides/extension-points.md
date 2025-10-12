# Extension Points

The betting stack is designed to be pluggable. This guide summarises the core interfaces for adding
new scrapers, models, and analytics pipelines.

## Scrapers

Implement `nflreadpy.betting.scrapers.base.SportsbookScraper` to onboard a new sportsbook or feed.
Subclasses override `_fetch_lines_impl()` and return a list of `OddsQuote` objects. The base class
handles retry, timeout, and polling cadence logic.

```python
from nflreadpy.betting.scrapers.base import OddsQuote, SportsbookScraper

class AcmeSportsbookScraper(SportsbookScraper):
    name = "acme"
    poll_interval_seconds = 10

    async def _fetch_lines_impl(self) -> list[OddsQuote]:
        payload = await self._call_api()
        return [self._quote_from_payload(item) for item in payload]
```

Register custom scrapers via the `[scrapers]` section of `config/betting.toml` or pass them directly
to `MultiScraperCoordinator`. Use the bundled `NameNormalizer` to map team names and player tokens
into canonical identifiers before analytics consume them.

## Models

Monte Carlo simulations run through `nflreadpy.betting.models.MonteCarloEngine`. Compose the engine
with custom `TeamRating` values or subclass it to provide alternative scoring distributions. Player
props plug into `PlayerPropForecaster`, which accepts arbitrary projection distributions per market.

To add a bespoke model:

1. Create a class that exposes `simulate_game(event_id, home_team, away_team)` and returns a
   `SimulationResult`.
2. Register the model in your configuration by pointing `analytics.model_factory` to a callable that
   returns the engine instance.
3. Provide calibration datasets by implementing `HistoricalGameRecord` builders or by feeding the
   engine with your own empirical priors.

The analytics scheduler loads the configured factory through `create_ingestion_service()` and
`create_edge_detector()`, so custom models propagate to the CLI, dashboards, and alerts without extra
plumbing.

## Analytics

Extend `nflreadpy.betting.analytics.EdgeDetector` or `LineMovementAnalyzer` to introduce new
analytics. Both classes accept dependency injection through the configuration helpers:

- `EdgeDetector` consumes simulated results and `OddsQuote` objects. Override `_evaluate_probabilities`
  or wrap the detector to incorporate bespoke risk adjustments.
- `LineMovementAnalyzer` ingests `IngestedOdds` records and produces movement summaries. Subclass it to
  emit custom metrics or to integrate with proprietary backtesting logic.

Analytics components surface via the CLI (`simulate`, `scan`, `backtest`), dashboards, and alerting
layer. When registering a new analytics pipeline:

1. Add the factory to `config/betting.toml` under `[analytics.pipelines]`.
2. Expose configuration toggles so operators can enable or disable the pipeline per environment.
3. Export additional telemetry using the metrics outlined in the [operations guide](operations.md).

## Packaging extensions

Ship reusable extensions as Python packages that depend on `nflreadpy[betting]`. Define an entry
point group (e.g. `nflreadpy.betting.scrapers`) to auto-discover plugins at runtime, or import the
modules in your deployment scripts before launching the ingestion workers.
