import asyncio
import datetime as dt

from nflreadpy.betting.ingestion import IngestedOdds, OddsIngestionService
from nflreadpy.betting.scrapers.base import (
    OddsQuote,
    SportsbookScraper,
    best_prices_by_selection,
)


def _collect_quotes(scrapers):
    async def _collect() -> list[OddsQuote]:
        batches = await asyncio.gather(
            *[scraper.fetch_lines() for scraper in scrapers]
        )
        return [quote for batch in batches for quote in batch]

    return asyncio.run(_collect())


def test_http_scrapers_fetch_quotes(http_scrapers):
    """Recorded payloads are normalised into OddsQuote objects."""

    quotes = _collect_quotes(http_scrapers)

    assert quotes
    assert {quote.sportsbook for quote in quotes} == {"fanduel", "draftkings", "pinnacle"}
    assert all(quote.observed_at.tzinfo is dt.timezone.utc for quote in quotes)
    assert all(isinstance(quote.american_odds, int) for quote in quotes)

    patriots = [
        quote
        for quote in quotes
        if quote.event_id == "2024-NE-NYJ"
        and quote.market == "moneyline"
        and quote.team_or_player == "NE"
    ]
    assert patriots
    assert {quote.american_odds for quote in patriots} == {-130, -125, -128}


def _ingested_to_quote(record: IngestedOdds) -> OddsQuote:
    return OddsQuote(
        event_id=record.event_id,
        sportsbook=record.sportsbook,
        book_market_group=record.book_market_group,
        market=record.market,
        scope=record.scope,
        entity_type=record.entity_type,
        team_or_player=record.team_or_player,
        side=record.side,
        line=record.line,
        american_odds=record.american_odds,
        observed_at=record.observed_at,
        extra=record.extra,
    )


def test_ingestion_rejects_stale_quotes(tmp_path, scraper_configs, alert_sink):
    service = OddsIngestionService(
        scrapers=None,
        scraper_configs=scraper_configs,
        storage_path=tmp_path / "stale.sqlite3",
        stale_after=dt.timedelta(minutes=1),
        alert_sink=alert_sink,
    )

    ingested = asyncio.run(service.fetch_and_store())
    assert ingested == []

    metrics = service.metrics
    assert "stale" in service.last_validation_summary
    assert metrics["persisted"] == 0
    assert metrics["requested"] > 0
    discarded = metrics["discarded"]
    assert sum(discarded.values()) == metrics["requested"]
    assert discarded.get("stale", 0) >= 1
    assert metrics["errors"]["validation"] == metrics["requested"]
    assert metrics["errors"]["scrapers"] == 0
    assert set(metrics["per_scraper"]) == {"fanduel", "draftkings", "pinnacle"}
    assert any(subject == "Odds validation issues" for subject, *_ in alert_sink.messages)


def test_ingestion_best_price_aggregation(tmp_path, http_scrapers):
    service = OddsIngestionService(
        scrapers=http_scrapers,
        storage_path=tmp_path / "quotes.sqlite3",
        stale_after=dt.timedelta(hours=1),
    )

    ingested = asyncio.run(service.fetch_and_store())
    assert ingested
    assert service.metrics["persisted"] == len(ingested)
    assert service.metrics["errors"]["scrapers"] == 0
    assert service.metrics["errors"]["validation"] == sum(
        service.metrics["discarded"].values()
    )

    quotes = [_ingested_to_quote(record) for record in ingested]
    best = best_prices_by_selection(quotes)

    ne_key = ("2024-NE-NYJ", "moneyline", "game", "NE", None, None)
    nyj_key = ("2024-NE-NYJ", "moneyline", "game", "NYJ", None, None)

    assert best[ne_key].sportsbook == "draftkings"
    assert best[ne_key].american_odds == -125
    assert best[nyj_key].sportsbook == "pinnacle"
    assert best[nyj_key].american_odds == 130


class BoomScraper(SportsbookScraper):
    name = "boom"

    def __init__(self) -> None:
        self.retry_attempts = 0

    async def _fetch_lines_impl(self) -> list[OddsQuote]:  # pragma: no cover - trivial
        raise RuntimeError("boom")


def test_scraper_failure_metrics_and_alerts(tmp_path, alert_sink) -> None:
    service = OddsIngestionService(
        scrapers=[BoomScraper()],
        storage_path=tmp_path / "boom.sqlite3",
        stale_after=dt.timedelta(minutes=5),
        alert_sink=alert_sink,
    )

    results = asyncio.run(service.fetch_and_store())
    assert results == []
    metrics = service.metrics
    assert metrics["errors"]["scrapers"] == 1
    assert metrics["errors"]["validation"] == 0
    assert metrics["per_scraper"]["boom"]["error"]
    assert any(subject == "Odds scraper failures" for subject, *_ in alert_sink.messages)
