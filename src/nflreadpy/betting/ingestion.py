"""Odds ingestion and persistence orchestration."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Sequence

from .normalization import NameNormalizer, default_normalizer
from .scrapers.base import MultiScraperCoordinator, OddsQuote, SportsbookScraper

logger = logging.getLogger(__name__)


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
        scrapers: Sequence[SportsbookScraper],
        storage_path: str | os.PathLike[str] = "betting_odds.sqlite3",
        normalizer: NameNormalizer | None = None,
        audit_logger: logging.Logger | None = None,
    ) -> None:
        self.scrapers = list(scrapers)
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._normalizer = normalizer or default_normalizer()
        self._coordinator = MultiScraperCoordinator(self.scrapers, self._normalizer)
        self._audit_logger = audit_logger or logging.getLogger("nflreadpy.betting.audit")
        self._init_db()

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

        quotes = await self._coordinator.collect_once()
        if not quotes:
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
            for quote in quotes
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

        logger.info("Stored %d odds quotes", len(payload))
        self._audit_logger.info(
            "ingestion.persisted",
            extra={
                "count": len(payload),
                "storage_path": str(self.storage_path),
            },
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
            for quote in quotes
        ]

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

