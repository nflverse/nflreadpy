"""FanDuel sportsbook scraper."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Iterable, List

from ..normalization import default_normalizer
from .base import OddsQuote, SportsbookScraper, normalise_american_odds
from .common import AsyncHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

_NORMALIZER = default_normalizer()


def _parse_timestamp(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        timestamp = dt.datetime.fromisoformat(value)
    except ValueError:
        logger.debug("Unable to parse timestamp %s", value)
        return dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("½", ".5"))
    except ValueError:
        return None


class FanDuelScraper(SportsbookScraper):
    """Concrete scraper for FanDuel's public odds APIs."""

    name = "fanduel"

    def __init__(
        self,
        endpoint: str,
        *,
        client: AsyncHTTPClient | None = None,
        rate_limit_per_second: float | None = 2.0,
        headers: Dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._endpoint = endpoint
        self._client = client or AsyncHTTPClient(timeout=self.timeout_seconds)
        self._rate_limiter = RateLimiter(rate_limit_per_second)
        self._headers = dict(headers or {})

    async def _fetch_lines_impl(self) -> List[OddsQuote]:
        await self._rate_limiter.wait()
        payload = await self._client.get_json(self._endpoint, headers=self._headers)
        quotes: List[OddsQuote] = []
        for event in payload.get("events", []):
            event_id = event.get("eventId") or event.get("id")
            if not event_id:
                continue
            observed_at = _parse_timestamp(event.get("lastUpdated"))
            markets: Iterable[Dict[str, Any]] = event.get("markets", [])
            for market in markets:
                group = market.get("group") or market.get("groupName") or "Other"
                market_key = market.get("key") or market.get("market") or "unknown"
                scope = market.get("scope") or "game"
                entity_type = market.get("entityType") or "team"
                for outcome in market.get("outcomes", []):
                    participant = (
                        outcome.get("participant")
                        or outcome.get("label")
                        or outcome.get("name")
                    )
                    if not participant:
                        continue
                    price_raw = (
                        outcome.get("price")
                        or outcome.get("odds")
                        or outcome.get("americanOdds")
                    )
                    if price_raw in {None, ""}:
                        continue
                    try:
                        price = normalise_american_odds(price_raw)
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("Skipping FanDuel outcome with price %s", price_raw)
                        continue
                    line = _safe_float(
                        outcome.get("line")
                        or outcome.get("points")
                        or outcome.get("handicap")
                    )
                    side = outcome.get("side") or outcome.get("designation")
                    extra: Dict[str, Any] = {}
                    if market_id := market.get("id"):
                        extra["market_id"] = market_id
                    if outcome_id := outcome.get("id"):
                        extra["outcome_id"] = outcome_id
                    if participants := outcome.get("participants"):
                        extra["participants"] = participants
                    canonical_participant = str(participant)
                    if entity_type == "team":
                        canonical_participant = _NORMALIZER.canonical_team(
                            canonical_participant
                        )
                    elif entity_type in {"player", "either", "leader"}:
                        canonical_participant = _NORMALIZER.canonical_player(
                            canonical_participant
                        )
                    if participants := outcome.get("participants"):
                        extra["participants"] = [
                            _NORMALIZER.canonical_player(str(p)) for p in participants
                        ]
                    quotes.append(
                        OddsQuote(
                            event_id=str(event_id),
                            sportsbook=self.name,
                            book_market_group=str(group),
                            market=str(market_key),
                            scope=str(scope),
                            entity_type=str(entity_type),
                            team_or_player=canonical_participant,
                            side=str(side) if side is not None else None,
                            line=line,
                            american_odds=price,
                            observed_at=observed_at,
                            extra=extra,
                        )
                    )
        logger.debug("FanDuel scraper produced %d quotes", len(quotes))
        return quotes
