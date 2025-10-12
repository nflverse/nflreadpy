import asyncio
import datetime as dt
from typing import Dict, Type

import pytest

from nflreadpy.betting.scrapers.base import (
    OddsQuote,
    SportsbookScraper,
    best_prices_by_selection,
)
from nflreadpy.betting.scrapers.draftkings import DraftKingsScraper
from nflreadpy.betting.scrapers.fanduel import FanDuelScraper
from nflreadpy.betting.scrapers.mock import MockSportsbookScraper
from nflreadpy.betting.scrapers.pinnacle import PinnacleScraper


@pytest.mark.parametrize(
    (
        "scraper_cls",
        "endpoint",
        "expected_prices",
    ),
    [
        (FanDuelScraper, "https://stub/fanduel", {"NE": -130, "NYJ": 120}),
        (DraftKingsScraper, "https://stub/draftkings", {"NE": -125, "NYJ": 122}),
        (PinnacleScraper, "https://stub/pinnacle", {"NE": -128, "NYJ": 130}),
    ],
)
def test_scraper_normalises_quotes(
    scraper_cls: Type[SportsbookScraper],
    endpoint: str,
    expected_prices: Dict[str, int],
    fresh_stub_client,
) -> None:
    scraper = scraper_cls(endpoint, client=fresh_stub_client, rate_limit_per_second=None)
    quotes = asyncio.run(scraper.fetch_lines())
    assert quotes
    assert {quote.sportsbook for quote in quotes} == {scraper.name}
    assert all(quote.observed_at.tzinfo is dt.timezone.utc for quote in quotes)

    moneyline_prices = {
        quote.team_or_player: quote.american_odds
        for quote in quotes
        if quote.market == "moneyline" and quote.entity_type == "team"
    }
    assert set(expected_prices).issubset(moneyline_prices)
    for team, price in expected_prices.items():
        assert moneyline_prices[team] == price

    # ensure player props are normalised via the shared name normaliser
    player_quotes = [
        quote
        for quote in quotes
        if quote.entity_type in {"player", "either", "leader"}
    ]
    if player_quotes:
        assert all(
            quote.team_or_player == quote.team_or_player.title()
            for quote in player_quotes
        )


def test_best_price_aggregation_across_books(fresh_stub_client) -> None:
    scrapers = [
        FanDuelScraper("https://stub/fanduel", client=fresh_stub_client, rate_limit_per_second=None),
        DraftKingsScraper(
            "https://stub/draftkings", client=fresh_stub_client, rate_limit_per_second=None
        ),
        PinnacleScraper(
            "https://stub/pinnacle", client=fresh_stub_client, rate_limit_per_second=None
        ),
    ]
    quotes: list[OddsQuote] = []
    for scraper in scrapers:
        quotes.extend(asyncio.run(scraper.fetch_lines()))

    best = best_prices_by_selection(quotes)
    ne_key = ("2024-NE-NYJ", "moneyline", "game", "NE", None, None)
    nyj_key = ("2024-NE-NYJ", "moneyline", "game", "NYJ", None, None)

    assert best[ne_key].sportsbook == "draftkings"
    assert best[ne_key].american_odds == -125
    assert best[nyj_key].sportsbook == "pinnacle"
    assert best[nyj_key].american_odds == 130


def test_mock_scraper_seed_is_deterministic() -> None:
    seeded_quotes = asyncio.run(MockSportsbookScraper(seed=123).fetch_lines())
    repeat_seed_quotes = asyncio.run(MockSportsbookScraper(seed=123).fetch_lines())
    comparable_fields = [
        (
            quote.event_id,
            quote.market,
            quote.scope,
            quote.entity_type,
            quote.team_or_player,
            quote.side,
            quote.line,
            quote.american_odds,
        )
        for quote in seeded_quotes
    ]
    repeat_fields = [
        (
            quote.event_id,
            quote.market,
            quote.scope,
            quote.entity_type,
            quote.team_or_player,
            quote.side,
            quote.line,
            quote.american_odds,
        )
        for quote in repeat_seed_quotes
    ]
    assert comparable_fields == repeat_fields

    moneylines = {
        (quote.event_id, quote.team_or_player): quote.american_odds
        for quote in seeded_quotes
        if quote.market == "moneyline" and quote.scope == "game"
    }
    assert moneylines == {
        ("2024-NE-NYJ", "NE"): -185,
        ("2024-NE-NYJ", "NYJ"): 185,
        ("2024-DEN-KC", "DEN"): -105,
        ("2024-DEN-KC", "KC"): 105,
        ("2024-BUF-MIA", "BUF"): -185,
        ("2024-BUF-MIA", "MIA"): 185,
        ("2024-PHI-DAL", "PHI"): -110,
        ("2024-PHI-DAL", "DAL"): 110,
        ("2024-DET-GB", "DET"): -150,
        ("2024-DET-GB", "GB"): 150,
    }
