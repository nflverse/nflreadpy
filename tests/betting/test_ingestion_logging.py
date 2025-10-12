import asyncio
import datetime as dt
import logging
from pathlib import Path

import pytest

from nflreadpy.betting.compliance import ComplianceConfig, ComplianceEngine
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


def test_injected_audit_logger_receives_events(tmp_path: Path) -> None:
    class ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            self.records.append(record)

    now = dt.datetime.now(dt.timezone.utc)
    quotes = [
        OddsQuote(
            event_id="E-custom",
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
    db_path = tmp_path / "custom_logger.sqlite3"
    custom_logger = logging.getLogger("test.ingestion.custom")
    handler = ListHandler()
    previous_handlers = list(custom_logger.handlers)
    previous_level = custom_logger.level
    previous_propagate = custom_logger.propagate
    for existing in previous_handlers:
        custom_logger.removeHandler(existing)
    custom_logger.addHandler(handler)
    custom_logger.setLevel(logging.INFO)
    custom_logger.propagate = False

    service = OddsIngestionService(
        scrapers=[StaticScraper("invalid", quotes)],
        storage_path=db_path,
        stale_after=dt.timedelta(minutes=5),
        audit_logger=custom_logger,
    )

    try:
        results = asyncio.run(service.fetch_and_store())
    finally:
        custom_logger.removeHandler(handler)
        for existing in previous_handlers:
            custom_logger.addHandler(existing)
        custom_logger.setLevel(previous_level)
        custom_logger.propagate = previous_propagate

    assert results == []
    assert any(record.getMessage() == "ingestion.discarded" for record in handler.records)
    assert all(record.name == "test.ingestion.custom" for record in handler.records)


def test_ingestion_enforces_compliance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audit_logger = logging.getLogger("test.ingestion.compliance")
    config = ComplianceConfig(jurisdiction_allowlist={"co"})
    engine = ComplianceEngine(config, audit_logger=audit_logger)
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
            american_odds=-110,
            observed_at=now,
            extra={"jurisdictions": ["ny"]},
        )
    ]
    db_path = tmp_path / "odds.sqlite3"
    service = OddsIngestionService(
        scrapers=[StaticScraper("noncompliant", quotes)],
        storage_path=db_path,
        stale_after=dt.timedelta(minutes=5),
        audit_logger=audit_logger,
        compliance_engine=engine,
    )

    caplog.set_level(logging.WARNING, logger="test.ingestion.compliance")
    results = asyncio.run(service.fetch_and_store())
    assert results == []
    assert any(
        rec.getMessage() == "compliance.violation" for rec in caplog.records
    )
    assert any(
        key.startswith("non_compliant") for key in service.last_validation_summary
    )
    discarded_records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "ingestion.discarded"
    ]
    assert any(getattr(rec, "compliance_reasons", None) for rec in discarded_records)
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
