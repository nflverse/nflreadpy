"""Odds ingestion and persistence orchestration."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .alerts import AlertSink
from .normalization import NameNormalizer, default_normalizer
from .scrapers.base import MultiScraperCoordinator, OddsQuote, SportsbookScraper
from .scrapers.draftkings import DraftKingsScraper
from .scrapers.fanduel import FanDuelScraper
from .scrapers.pinnacle import PinnacleScraper

logger = logging.getLogger(__name__)


SCRAPER_REGISTRY: Mapping[str, type[SportsbookScraper]] = {
    "fanduel": FanDuelScraper,
    "draftkings": DraftKingsScraper,
    "pinnacle": PinnacleScraper,
}


@dataclasses.dataclass(slots=True)
class IngestedOdds:
    event_id: str
    sportsbook: str
    book_market_group: str
    market: str
    scope: str
    entity_type: str
    team_or_player: str
    side: str | None
    line: float | None
    american_odds: int
    observed_at: dt.datetime
    extra: dict[str, object]


class OddsIngestionService:
    """Coordinate scrapers, persistence, and consumer notifications."""

    def __init__(
        self,
        scrapers: Sequence[SportsbookScraper] | None = None,
        *,
        scraper_configs: Sequence[Mapping[str, Any]] | None = None,
        storage_path: str | os.PathLike[str] = "betting_odds.sqlite3",
        normalizer: NameNormalizer | None = None,
        stale_after: dt.timedelta = dt.timedelta(minutes=10),
        alert_sink: AlertSink | None = None,
    ) -> None:
        self.scrapers = list(scrapers or [])
        self.scrapers.extend(self._instantiate_scrapers(scraper_configs))
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._normalizer = normalizer or default_normalizer()
        self._coordinator = MultiScraperCoordinator(self.scrapers, self._normalizer)
        self._stale_after = stale_after
        self._alert_sink = alert_sink
        self._metrics: Dict[str, Any] = {
            "requested": 0,
            "persisted": 0,
            "discarded": {},
            "latency_seconds": 0.0,
            "per_scraper": {},
        }
        self._last_validation_summary: Dict[str, int] = {}
        self._init_db()

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self._metrics

    def _instantiate_scrapers(
        self, scraper_configs: Sequence[Mapping[str, Any]] | None
    ) -> List[SportsbookScraper]:
        if not scraper_configs:
            return []
        instances: List[SportsbookScraper] = []
        for config in scraper_configs:
            scraper_type = config.get("type")
            if not scraper_type:
                raise ValueError("Scraper config missing 'type'")
            scraper_cls = SCRAPER_REGISTRY.get(str(scraper_type).lower())
            if not scraper_cls:
                raise ValueError(f"Unknown scraper type: {scraper_type}")
            kwargs = {key: value for key, value in config.items() if key != "type"}
            instances.append(scraper_cls(**kwargs))
        return instances

    def _init_db(self) -> None:
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_quotes (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    sportsbook TEXT NOT NULL,
                    book_market_group TEXT NOT NULL,
                    market TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    team_or_player TEXT NOT NULL,
                    side TEXT,
                    line REAL,
                    side_key TEXT NOT NULL,
                    line_key TEXT NOT NULL,
                    american_odds INTEGER NOT NULL,
                    extra TEXT,
                    observed_at TEXT NOT NULL,
                    UNIQUE (
                        event_id,
                        sportsbook,
                        book_market_group,
                        market,
                        scope,
                        entity_type,
                        team_or_player,
                        side_key,
                        line_key
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_quotes_history (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    sportsbook TEXT NOT NULL,
                    book_market_group TEXT NOT NULL,
                    market TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    team_or_player TEXT NOT NULL,
                    side TEXT,
                    line REAL,
                    side_key TEXT NOT NULL,
                    line_key TEXT NOT NULL,
                    american_odds INTEGER NOT NULL,
                    extra TEXT,
                    observed_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    async def fetch_and_store(self) -> List[IngestedOdds]:
        """Fetch odds from all scrapers and persist them."""

        start = time.perf_counter()
        quotes = await self._coordinator.collect_once()
        elapsed = time.perf_counter() - start
        per_scraper = {
            name: dict(details)
            for name, details in self._coordinator.last_run_details.items()
        }
        self._maybe_alert_on_scraper_failures(per_scraper)
        if not quotes:
            self._metrics = {
                "requested": 0,
                "persisted": 0,
                "discarded": {},
                "latency_seconds": elapsed,
                "per_scraper": per_scraper,
            }
            return []

        valid_quotes = self._filter_valid_quotes(quotes)

        if not valid_quotes:
            logger.info("All %d quotes discarded during validation", len(quotes))
            self._metrics = {
                "requested": len(quotes),
                "persisted": 0,
                "discarded": {"validation_failed": len(quotes)},
                "latency_seconds": elapsed,
                "per_scraper": per_scraper,
            }
            self._emit_validation_alert(len(quotes), self._last_validation_summary)
            return []

        payload = [
            (
                quote.event_id,
                quote.sportsbook,
                quote.book_market_group,
                quote.market,
                quote.scope,
                quote.entity_type,
                quote.team_or_player,
                quote.side,
                quote.line,
                (quote.side or ""),
                "" if quote.line is None else f"{quote.line:.4f}",
                quote.american_odds,
                json.dumps(quote.extra or {}, sort_keys=True),
                quote.observed_at.isoformat(),
            )
            for quote in valid_quotes
        ]

        with sqlite3.connect(self.storage_path) as conn:
            conn.executemany(
                """
                INSERT INTO odds_quotes(
                    event_id,
                    sportsbook,
                    book_market_group,
                    market,
                    scope,
                    entity_type,
                    team_or_player,
                    side,
                    line,
                    side_key,
                    line_key,
                    american_odds,
                    extra,
                    observed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    event_id,
                    sportsbook,
                    book_market_group,
                    market,
                    scope,
                    entity_type,
                    team_or_player,
                    side_key,
                    line_key
                ) DO UPDATE SET
                    american_odds=excluded.american_odds,
                    extra=excluded.extra,
                    observed_at=excluded.observed_at
                """,
                payload,
            )
            conn.executemany(
                """
                INSERT INTO odds_quotes_history(
                    event_id,
                    sportsbook,
                    book_market_group,
                    market,
                    scope,
                    entity_type,
                    team_or_player,
                    side,
                    line,
                    side_key,
                    line_key,
                    american_odds,
                    extra,
                    observed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()

        discarded_summary = self._last_validation_summary
        self._metrics = {
            "requested": len(quotes),
            "persisted": len(valid_quotes),
            "discarded": discarded_summary,
            "latency_seconds": elapsed,
            "per_scraper": per_scraper,
        }
        if discarded_summary:
            self._emit_validation_alert(len(quotes), discarded_summary)
        logger.info(
            "Stored %d odds quotes (%d discarded: %s)",
            len(valid_quotes),
            len(quotes) - len(valid_quotes),
            discarded_summary,
        )
        return [
            IngestedOdds(
                event_id=quote.event_id,
                sportsbook=quote.sportsbook,
                book_market_group=quote.book_market_group,
                market=quote.market,
                scope=quote.scope,
                entity_type=quote.entity_type,
                team_or_player=quote.team_or_player,
                side=quote.side,
                line=quote.line,
                american_odds=quote.american_odds,
                observed_at=quote.observed_at,
                extra=dict(quote.extra or {}),
            )
            for quote in valid_quotes
        ]

    def _filter_valid_quotes(self, quotes: Iterable[OddsQuote]) -> List[OddsQuote]:
        now = dt.datetime.now(dt.timezone.utc)
        valid: List[OddsQuote] = []
        summary: MutableMapping[str, int] = {}
        for quote in quotes:
            reason = self._validate_quote(quote, now)
            if reason is None:
                valid.append(quote)
            else:
                summary[reason] = summary.get(reason, 0) + 1
                logger.debug(
                    "Discarding quote from %s/%s (%s): %s",
                    quote.sportsbook,
                    quote.market,
                    quote.event_id,
                    reason,
                )
        self._last_validation_summary = dict(summary)
        if summary:
            logger.warning("Discarded quotes summary: %s", summary)
        return valid

    def _validate_quote(
        self, quote: OddsQuote, reference_time: dt.datetime
    ) -> str | None:
        if quote.american_odds == 0:
            return "invalid_odds"
        if not isinstance(quote.american_odds, int):
            return "invalid_odds"
        if abs(quote.american_odds) > 100000:
            return "invalid_odds"
        if abs(quote.american_odds) < 100 and quote.american_odds not in {100, -100}:
            return "invalid_odds"
        if quote.line is not None and not isinstance(quote.line, (int, float)):
            return "invalid_line"
        if isinstance(quote.line, float) and not math.isfinite(quote.line):
            return "invalid_line"
        if not quote.team_or_player:
            return "missing_selection"
        observed = quote.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=dt.timezone.utc)
        if observed < reference_time - self._stale_after:
            return "stale"
        return None

    def _maybe_alert_on_scraper_failures(
        self, per_scraper: Mapping[str, Mapping[str, Any]]
    ) -> None:
        failures = {
            name: details.get("error")
            for name, details in per_scraper.items()
            if details.get("error")
        }
        if not failures:
            return
        logger.error("Scraper failures detected: %s", failures)
        self._send_alert(
            "Odds scraper failures",
            "One or more sportsbook scrapers failed during collection.",
            metadata={"failures": failures},
        )

    def _emit_validation_alert(
        self, requested: int, summary: Mapping[str, int]
    ) -> None:
        if not summary:
            return
        logger.warning("Validation rejected %d quotes: %s", requested, summary)
        self._send_alert(
            "Odds validation issues",
            f"Validation rejected {sum(summary.values())} of {requested} quotes.",
            metadata={"discarded": dict(summary)},
        )

    def _send_alert(
        self,
        subject: str,
        body: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._alert_sink:
            return
        try:
            self._alert_sink.send(subject, body, metadata=metadata)
        except Exception:  # pragma: no cover - alert sinks shouldn't break ingestion
            logger.exception("Failed to dispatch alert: %s", subject)

    def load_latest(self, event_id: str | None = None) -> List[IngestedOdds]:
        """Load the latest stored odds from disk."""

        query = (
            "SELECT event_id, sportsbook, book_market_group, market, scope, "
            "entity_type, team_or_player, side, line, american_odds, extra, observed_at "
            "FROM odds_quotes"
        )
        params: tuple[str, ...] = ()
        if event_id:
            query += " WHERE event_id = ?"
            params = (event_id,)
        with sqlite3.connect(self.storage_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            IngestedOdds(
                event_id=row[0],
                sportsbook=row[1],
                book_market_group=row[2],
                market=row[3],
                scope=row[4],
                entity_type=row[5],
                team_or_player=row[6],
                side=row[7],
                line=float(row[8]) if row[8] is not None else None,
                american_odds=int(row[9]),
                extra=json.loads(row[10] or "{}"),
                observed_at=dt.datetime.fromisoformat(row[11]),
            )
            for row in rows
        ]

    def load_history(
        self,
        event_id: str | None = None,
        limit: int | None = None,
    ) -> List[IngestedOdds]:
        """Load historical odds snapshots for movement analysis."""

        query = (
            "SELECT event_id, sportsbook, book_market_group, market, scope, "
            "entity_type, team_or_player, side, line, american_odds, extra, observed_at "
            "FROM odds_quotes_history"
        )
        params: list[object] = []
        if event_id:
            query += " WHERE event_id = ?"
            params.append(event_id)
        query += " ORDER BY observed_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with sqlite3.connect(self.storage_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            IngestedOdds(
                event_id=row[0],
                sportsbook=row[1],
                book_market_group=row[2],
                market=row[3],
                scope=row[4],
                entity_type=row[5],
                team_or_player=row[6],
                side=row[7],
                line=float(row[8]) if row[8] is not None else None,
                american_odds=int(row[9]),
                extra=json.loads(row[10] or "{}"),
                observed_at=dt.datetime.fromisoformat(row[11]),
            )
            for row in rows
        ]

    async def run_forever(self, interval_seconds: float = 30.0) -> None:
        """Continuously collect odds using the configured scrapers."""

        while True:
            await self.fetch_and_store()
            await asyncio.sleep(interval_seconds)

