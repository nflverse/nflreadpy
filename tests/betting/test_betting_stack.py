from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import List

import pytest

from nflreadpy.betting.analytics import EdgeDetector
from nflreadpy.betting.dashboard import Dashboard
from nflreadpy.betting.ingestion import IngestedOdds, OddsIngestionService
from nflreadpy.betting.models import (
    GameSimulationConfig,
    MonteCarloEngine,
    PlayerProjection,
    PlayerPropForecaster,
    SimulationResult,
    TeamRating,
)
from nflreadpy.betting.scrapers.base import OddsQuote, StaticScraper


@pytest.fixture()
def now() -> dt.datetime:
    return dt.datetime(2024, 9, 1, 12, tzinfo=dt.timezone.utc)


@pytest.fixture()
def static_quotes(now: dt.datetime) -> List[OddsQuote]:
    return [
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side=None,
            line=None,
            american_odds=-135,
            observed_at=now,
            extra={"opponent": "NYJ"},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NYJ",
            side=None,
            line=None,
            american_odds=+125,
            observed_at=now,
            extra={"opponent": "NE"},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side="fav",
            line=-3.5,
            american_odds=-105,
            observed_at=now,
            extra={},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="total",
            scope="game",
            entity_type="total",
            team_or_player="Total",
            side="over",
            line=41.5,
            american_odds=-108,
            observed_at=now,
            extra={},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Player Props",
            market="receiving_yards",
            scope="game",
            entity_type="player",
            team_or_player="Garrett Wilson",
            side="over",
            line=68.5,
            american_odds=+120,
            observed_at=now,
            extra={"projection_mean": 82.0, "projection_stdev": 16.0},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Either Player",
            market="longest_reception",
            scope="game",
            entity_type="either",
            team_or_player="Sutton/Dobbins",
            side="yes",
            line=29.5,
            american_odds=+140,
            observed_at=now,
            extra={"participants": ["Courtland Sutton", "J.K. Dobbins"], "projection_mean": 0.55},
        ),
    ]


@pytest.fixture()
def static_scraper(static_quotes: List[OddsQuote]) -> StaticScraper:
    return StaticScraper("testbook", static_quotes)


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "odds.sqlite3"


@pytest.fixture()
def ingestion(static_scraper: StaticScraper, tmp_db_path: Path) -> OddsIngestionService:
    return OddsIngestionService([static_scraper], storage_path=tmp_db_path)


@pytest.fixture()
def engine() -> MonteCarloEngine:
    ratings = {
        "NE": TeamRating(team="NE", offensive_rating=1.5, defensive_rating=0.6),
        "NYJ": TeamRating(team="NYJ", offensive_rating=-0.2, defensive_rating=0.3),
    }
    return MonteCarloEngine(ratings, GameSimulationConfig(iterations=1_000, seed=42))


@pytest.fixture()
def simulation(engine: MonteCarloEngine) -> SimulationResult:
    return engine.simulate_game("2024-NE-NYJ", "NE", "NYJ")


@pytest.fixture()
def player_model() -> PlayerPropForecaster:
    model = PlayerPropForecaster(
        [
            PlayerProjection(
                player="Courtland Sutton",
                market="longest_reception",
                mean=0.28,
                stdev=0.1,
                distribution="bernoulli",
            ),
            PlayerProjection(
                player="J.K. Dobbins",
                market="longest_reception",
                mean=0.32,
                stdev=0.1,
                distribution="bernoulli",
            ),
        ]
    )
    return model


def test_ingestion_fetch_and_store(ingestion: OddsIngestionService, static_quotes: List[OddsQuote]) -> None:
    stored = asyncio.run(ingestion.fetch_and_store())
    assert len(stored) == len(static_quotes)
    loaded = ingestion.load_latest("2024-NE-NYJ")
    assert {row.team_or_player for row in loaded if row.market == "moneyline"} == {"NE", "NYJ"}
    spread_quote = next(row for row in loaded if row.market == "spread" and row.team_or_player == "NE")
    assert spread_quote.line == -3.5
    assert spread_quote.book_market_group == "Game Lines"


def test_simulation_supports_multiple_markets(simulation: SimulationResult) -> None:
    moneyline = simulation.moneyline_probability("NE")
    assert 0.0 < moneyline < 1.0
    spread = simulation.spread_probability("NE", -6.5)
    assert 0.0 <= spread.win <= 1.0
    total = simulation.total_probability("over", 41.5)
    assert 0.0 <= total.win <= 1.0
    team_total = simulation.team_total_probability("NE", "over", 23.5)
    assert 0.0 <= team_total.win <= 1.0


def test_edge_detector_identifies_varied_edges(
    static_quotes: List[OddsQuote],
    simulation: SimulationResult,
    player_model: PlayerPropForecaster,
) -> None:
    detector = EdgeDetector(value_threshold=0.0, player_model=player_model)
    opportunities = detector.detect(static_quotes, [simulation])
    markets = {opp.market for opp in opportunities}
    assert {"spread", "total", "receiving_yards", "longest_reception"}.issubset(markets)
    for opp in opportunities:
        assert opp.expected_value >= 0.0
        assert opp.kelly_fraction >= 0.0


def test_dashboard_renders_sections(
    static_quotes: List[OddsQuote],
    simulation: SimulationResult,
    player_model: PlayerPropForecaster,
) -> None:
    detector = EdgeDetector(value_threshold=0.0, player_model=player_model)
    opportunities = detector.detect(static_quotes, [simulation])
    ingested = [
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
        for quote in static_quotes
    ]
    dashboard = Dashboard()
    rendered = dashboard.render(ingested, [simulation], opportunities)
    assert "Simulations" in rendered
    assert "Latest Quotes" in rendered
    assert "Opportunities" in rendered
