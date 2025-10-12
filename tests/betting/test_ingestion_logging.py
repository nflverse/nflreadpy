import asyncio
import datetime as dt
import logging
from pathlib import Path

import pytest

from nflreadpy.betting.ingestion import OddsIngestionService
from nflreadpy.betting.scrapers.base import OddsQuote, StaticScraper


def test_ingestion_logs_and_captures_invalid_quotes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audit_logger = logging.getLogger("test.ingestion")
    now = dt.datetime.now(dt.timezone.utc)
    quotes = [
        OddsQuote(
            event_id="E1",
            sportsbook="book",
            book_market_group="Game Lines",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side="fav",
            line=-3.5,
            american_odds=0,
            observed_at=now,
            extra={"market_rules": {"push_handling": "push"}},
        )
    ]
    db_path = tmp_path / "odds.sqlite3"
    service = OddsIngestionService(
        scrapers=[StaticScraper("invalid", quotes)],
        storage_path=db_path,
        stale_after=dt.timedelta(minutes=5),
        audit_logger=audit_logger,
    )

    caplog.set_level(logging.WARNING, logger="test.ingestion")
    results = asyncio.run(service.fetch_and_store())
    assert results == []

    discarded_logs = [
        rec for rec in caplog.records if rec.getMessage() == "ingestion.discarded"
    ]
    assert discarded_logs
    summary_logs = [
        rec for rec in caplog.records if rec.getMessage() == "ingestion.validation_failed"
    ]
    assert summary_logs
    assert service.metrics["persisted"] == 0
    assert service.last_validation_summary.get("invalid_odds") == 1


def test_future_timestamp_rejected(tmp_path: Path) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    future_quote = OddsQuote(
        event_id="E2",
        sportsbook="book",
        book_market_group="Game Lines",
        market="total",
        scope="game",
        entity_type="total",
        team_or_player="Total",
        side="over",
        line=41.5,
        american_odds=-110,
        observed_at=now + dt.timedelta(hours=1),
        extra={},
    )
    db_path = tmp_path / "future.sqlite3"
    service = OddsIngestionService(
        scrapers=[StaticScraper("future", [future_quote])],
        storage_path=db_path,
        stale_after=dt.timedelta(minutes=5),
    )

    results = asyncio.run(service.fetch_and_store())
    assert results == []
    assert service.last_validation_summary.get("future_timestamp") == 1
    assert service.metrics["discarded"].get("future_timestamp") == 1
