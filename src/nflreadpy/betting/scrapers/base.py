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
from abc import ABC, abstractmethod
from fractions import Fraction
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


def normalise_american_odds(value: int | float | str) -> int:
    """Coerce American odds into a signed integer.

    Sportsbooks sometimes omit the ``+`` sign on positive numbers or expose
    odds as strings.  Downstream analytics expect a canonical integer
    representation, so we normalise here.
    """

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    stripped = value.strip()
    if not stripped:
        raise ValueError("Empty odds value")
    if stripped[0] in {"+", "-"}:
        return int(stripped)
    return int(f"+{stripped}")


def american_to_decimal(value: int | float | str) -> float:
    """Convert an American price into European decimal odds."""

    price = normalise_american_odds(value)
    if price == 0:
        raise ValueError("American odds cannot be zero")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / -price


def american_to_profit_multiplier(value: int | float | str) -> float:
    """Return the net profit multiplier for a one-unit stake at American odds."""

    price = normalise_american_odds(value)
    if price == 0:
        raise ValueError("American odds cannot be zero")
    if price > 0:
        return price / 100.0
    return 100.0 / -price


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to their American representation."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def fractional_to_decimal(numerator: int, denominator: int) -> float:
    """Convert fractional odds to decimal form."""

    if denominator == 0:
        raise ValueError("Fractional denominator cannot be zero")
    if numerator < 0 or denominator < 0:
        raise ValueError("Fractional odds must be non-negative")
    fraction = Fraction(numerator, denominator)
    return 1.0 + float(fraction)


def decimal_to_fractional(decimal_odds: float, *, max_denominator: int = 512) -> Tuple[int, int]:
    """Convert decimal odds to a simplified fractional representation."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    fraction = Fraction(decimal_odds - 1.0).limit_denominator(max_denominator)
    return fraction.numerator, fraction.denominator


def american_to_fractional(value: int | float | str, *, max_denominator: int = 512) -> Tuple[int, int]:
    """Convert American odds to fractional form."""

    decimal = american_to_decimal(value)
    return decimal_to_fractional(decimal, max_denominator=max_denominator)


def fractional_to_american(numerator: int, denominator: int) -> int:
    """Convert fractional odds to American format."""

    decimal = fractional_to_decimal(numerator, denominator)
    return decimal_to_american(decimal)


def implied_probability_from_decimal(decimal_odds: float) -> float:
    """Return the bookmaker's implied win probability from decimal odds."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    return 1.0 / decimal_odds


def implied_probability_from_fractional(numerator: int, denominator: int) -> float:
    """Return the implied probability from fractional odds."""

    decimal = fractional_to_decimal(numerator, denominator)
    return implied_probability_from_decimal(decimal)


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

    def implied_probability(self) -> float:
        """Convert American odds into decimal implied probability."""

        return implied_probability_from_decimal(american_to_decimal(self.american_odds))

    def decimal_multiplier(self) -> float:
        """Return the net profit multiplier for a one-unit stake."""

        return american_to_profit_multiplier(self.american_odds)


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

    async def collect_once(self) -> List[OddsQuote]:
        if not self.scrapers:
            return []
        tasks = [asyncio.create_task(scraper.fetch_lines()) for scraper in self.scrapers]
        results: List[OddsQuote] = []
        for task in tasks:
            try:
                quotes = await task
            except Exception as err:  # pragma: no cover - defensive
                logger.error("Scraper failure: %s", err)
                continue
            for quote in quotes:
                results.append(self.normalizer.normalise_quote(quote))
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


