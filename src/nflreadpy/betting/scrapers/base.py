"""Abstract sportsbook scraping interfaces.

The Bloomberg-style toolkit needs resilient, asynchronous scrapers that can
harvest prices from multiple operators.  To keep things testable without
hitting real sportsbooks, this module defines a small framework with clear
contracts and rich logging hooks.  Compared with the initial skeleton this
version expands the quote representation so that complex markets (alternate
spreads, player props, scope splits, etc.) can be expressed in a uniform
shape.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import logging
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

from ..normalization import NameNormalizer, default_normalizer
from ..utils import (
    OddsValue,
    american_to_decimal,
    american_to_fractional,
    american_to_profit_multiplier,
    decimal_to_american,
    decimal_to_fractional,
    fractional_to_american,
    fractional_to_decimal,
    implied_probability_from_american,
    implied_probability_from_decimal,
    implied_probability_from_fractional,
    implied_probability_to_american,
    implied_probability_to_decimal,
    implied_probability_to_fraction,
    normalise_american_odds,
)

logger = logging.getLogger(__name__)

ScopeLiteral = Literal[
    "game",
    "1h",
    "2h",
    "1q",
    "2q",
    "3q",
    "4q",
]
EntityLiteral = Literal["team", "player", "total", "either", "leader"]


@dataclasses.dataclass(slots=True)
class OddsQuote:
    """Represents a sportsbook quote in the canonical prompt-prescribed shape."""

    event_id: str
    sportsbook: str
    book_market_group: str
    market: str
    scope: ScopeLiteral
    entity_type: EntityLiteral
    team_or_player: str
    side: str | None
    line: float | None
    american_odds: int
    observed_at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    extra: Mapping[str, Any] | None = None

    def decimal_odds(self) -> float:
        """Return decimal odds for this quote."""

        return american_to_decimal(self.american_odds)

    def fractional_odds(self, *, max_denominator: int = 512) -> Tuple[int, int]:
        """Return fractional odds for this quote."""

        return american_to_fractional(
            self.american_odds, max_denominator=max_denominator
        )

    def profit_multiplier(self) -> float:
        """Return the net profit multiplier for a one-unit stake."""

        return american_to_profit_multiplier(self.american_odds)

    def implied_probability(self) -> float:
        """Convert American odds into decimal implied probability."""

        return implied_probability_from_american(self.american_odds)

    def decimal_multiplier(self) -> float:
        """Return the net profit multiplier for a one-unit stake."""

        return self.profit_multiplier()


class SportsbookScraper(ABC):
    """Base class for sportsbook scrapers with retry logic."""

    name: str = "generic"
    retry_attempts: int = 2
    retry_backoff: float = 0.5
    timeout_seconds: float = 10.0
    poll_interval_seconds: float = 60.0

    async def fetch_lines(self) -> List[OddsQuote]:
        """Fetch odds quotes with retry and timeout handling."""

        for attempt in range(1, self.retry_attempts + 2):
            try:
                return await asyncio.wait_for(
                    self._fetch_lines_impl(), timeout=self.timeout_seconds
                )
            except Exception as err:  # pragma: no cover - defensive logging
                logger.warning(
                    "Scraper %s attempt %s/%s failed: %s",
                    self.name,
                    attempt,
                    self.retry_attempts + 1,
                    err,
                )
                if attempt > self.retry_attempts:
                    raise
                await asyncio.sleep(self.retry_backoff * attempt)
        return []

    @abstractmethod
    async def _fetch_lines_impl(self) -> List[OddsQuote]:
        """Implementation hook for subclasses."""

    async def stream(self) -> Iterator[OddsQuote]:
        """Yield odds quotes indefinitely."""

        while True:
            lines = await self.fetch_lines()
            for line in lines:
                yield line
            await asyncio.sleep(self.poll_interval_seconds)


class MultiScraperCoordinator:
    """Run multiple scrapers concurrently with optional normalisation."""

    def __init__(
        self,
        scrapers: Sequence[SportsbookScraper],
        normalizer: NameNormalizer | None = None,
    ) -> None:
        self.scrapers = list(scrapers)
        self.normalizer = normalizer or default_normalizer()
        self._last_run_details: Dict[str, Dict[str, Any]] = {}

    @property
    def last_run_details(self) -> Mapping[str, Dict[str, Any]]:
        """Return diagnostics from the most recent collection."""

        return self._last_run_details

    async def _run_scraper(
        self, scraper: SportsbookScraper
    ) -> Tuple[List[OddsQuote], Exception | None, float]:
        start = time.perf_counter()
        try:
            quotes = await scraper.fetch_lines()
            error: Exception | None = None
        except Exception as err:  # pragma: no cover - defensive surface area
            quotes = []
            error = err
        elapsed = time.perf_counter() - start
        return quotes, error, elapsed

    async def collect_once(self) -> List[OddsQuote]:
        if not self.scrapers:
            return []
        tasks = {
            asyncio.create_task(self._run_scraper(scraper)): scraper for scraper in self.scrapers
        }
        diagnostics: Dict[str, Dict[str, Any]] = {}
        results: List[OddsQuote] = []
        for task, scraper in tasks.items():
            try:
                quotes, error, elapsed = await task
            except Exception as err:  # pragma: no cover - extremely defensive
                logger.exception("Scraper %s raised unexpectedly", scraper.name)
                diagnostics[scraper.name] = {
                    "count": 0,
                    "latency_seconds": 0.0,
                    "error": str(err),
                }
                continue
            diagnostics[scraper.name] = {
                "count": len(quotes),
                "latency_seconds": elapsed,
                "error": str(error) if error else None,
            }
            if error:
                logger.error("Scraper %s failed: %s", scraper.name, error)
                continue
            for quote in quotes:
                results.append(self.normalizer.normalise_quote(quote))
        self._last_run_details = diagnostics
        return results

    async def collect_stream(self) -> Iterator[OddsQuote]:
        while True:
            for quote in await self.collect_once():
                yield quote
            await asyncio.sleep(0)


class StaticScraper(SportsbookScraper):
    """Deterministic scraper used in tests and local development."""

    def __init__(self, name: str, payload: Iterable[OddsQuote]):
        self.name = name
        self._payload = list(payload)
        self.retry_attempts = 0
        self.timeout_seconds = 1.0
        self.poll_interval_seconds = 1.0

    async def _fetch_lines_impl(self) -> List[OddsQuote]:
        logger.debug(
            "Static scraper %s returning %d quotes", self.name, len(self._payload)
        )
        return [dataclasses.replace(line) for line in self._payload]


def aggregate_quotes(lines: Iterable[OddsQuote]) -> Dict[str, MutableMapping[str, List[OddsQuote]]]:
    """Group quotes by event then by market for downstream consumers."""

    aggregated: Dict[str, MutableMapping[str, List[OddsQuote]]] = {}
    for line in lines:
        aggregated.setdefault(line.event_id, {}).setdefault(line.market, []).append(line)
    return aggregated


def best_prices_by_selection(
    quotes: Iterable[OddsQuote],
) -> Dict[Tuple[str, str, str, str, str | None, float | None], OddsQuote]:
    """Return the best available price for each unique selection across books."""

    best: Dict[Tuple[str, str, str, str, str | None, float | None], OddsQuote] = {}
    for quote in quotes:
        key = (
            quote.event_id,
            quote.market,
            quote.scope,
            quote.team_or_player,
            quote.side,
            quote.line,
        )
        current = best.get(key)
        if current is None:
            best[key] = quote
            continue
        if quote.american_odds > current.american_odds:
            best[key] = quote
    return best


