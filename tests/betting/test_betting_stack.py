from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nflreadpy.betting.analytics import EdgeDetector
from nflreadpy.betting.dashboard import Dashboard
from nflreadpy.betting.ingestion import IngestedOdds, OddsIngestionService
from nflreadpy.betting.models import (
    GameSimulationConfig,
    MonteCarloEngine,
    PlayerFeatureRow,
    PlayerProjection,
    PlayerPropForecaster,
    SimulationResult,
    TeamRating,
)
from nflreadpy.betting.scrapers.base import (
    MultiScraperCoordinator,
    OddsQuote,
    SportsbookScraper,
    best_prices_by_selection,
)


def test_http_scrapers_normalise_and_best_price(
    http_scrapers: List[SportsbookScraper],
) -> None:
    coordinator = MultiScraperCoordinator(http_scrapers)
    quotes = asyncio.run(coordinator.collect_once())
    assert quotes
    diagnostics = coordinator.last_run_details
    assert set(diagnostics) == {"fanduel", "draftkings", "pinnacle"}
    assert all("latency_seconds" in info for info in diagnostics.values())
    assert all(info.get("error") is None for info in diagnostics.values())
    teams = {
        quote.team_or_player
        for quote in quotes
        if quote.market == "moneyline" and quote.entity_type == "team"
    }
    assert {"NE", "NYJ"}.issubset(teams)
    best = best_prices_by_selection(quotes)
    ne_key = ("2024-NE-NYJ", "moneyline", "game", "NE", None, None)
    nyj_key = ("2024-NE-NYJ", "moneyline", "game", "NYJ", None, None)
    assert best[ne_key].american_odds == -125
    assert best[nyj_key].american_odds == 130
    player_quote = next(quote for quote in quotes if quote.entity_type == "player")
    assert player_quote.team_or_player == "Garrett Wilson"


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


def test_bivariate_engine_uses_historical_records() -> None:
    ratings = {
        "NE": TeamRating(team="NE", offensive_rating=1.2, defensive_rating=0.4),
        "NYJ": TeamRating(team="NYJ", offensive_rating=-0.3, defensive_rating=0.7),
    }
    history = [
        {
            "home_team": "NE",
            "away_team": "NYJ",
            "home_points": 27,
            "away_points": 17,
            "home_pace": 65.0,
            "away_pace": 63.0,
            "home_offense_rating": 1.1,
            "home_defense_rating": 0.5,
            "away_offense_rating": -0.4,
            "away_defense_rating": 0.8,
        },
        {
            "home_team": "NYJ",
            "away_team": "NE",
            "home_points": 20,
            "away_points": 24,
            "home_pace": 60.0,
            "away_pace": 62.0,
            "home_offense_rating": -0.2,
            "home_defense_rating": 0.9,
            "away_offense_rating": 1.0,
            "away_defense_rating": 0.4,
        },
    ]
    engine = MonteCarloEngine(ratings, GameSimulationConfig(seed=101), historical_games=history)
    params = engine.calibrator.estimate_parameters(ratings["NE"], ratings["NYJ"])
    assert params.lambda_home > 20.0
    assert params.lambda_away > 15.0
    result = engine.simulate_game("historical", "NE", "NYJ")
    assert result.expected_total == pytest.approx(params.lambda_home + params.lambda_away, rel=0.05)


def test_player_pipelines_fit_and_covariance() -> None:
    rows = [
        PlayerFeatureRow(
            player="Courtland Sutton",
            opponent="NYJ",
            market="receiving_yards",
            scope="game",
            target=84.0,
            injury_status=0.1,
            weather=0.0,
            pace=65.0,
            weight=1.0,
            game_id="2023-NE-NYJ",
        ),
        PlayerFeatureRow(
            player="J.K. Dobbins",
            opponent="NYJ",
            market="receiving_yards",
            scope="game",
            target=28.0,
            injury_status=0.0,
            weather=0.0,
            pace=65.0,
            weight=1.0,
            game_id="2023-NE-NYJ",
        ),
        PlayerFeatureRow(
            player="Courtland Sutton",
            opponent="BUF",
            market="receiving_yards",
            scope="game",
            target=92.0,
            injury_status=0.2,
            weather=-5.0,
            pace=61.0,
            weight=0.9,
            game_id="2023-DEN-BUF",
        ),
        PlayerFeatureRow(
            player="J.K. Dobbins",
            opponent="BUF",
            market="receiving_yards",
            scope="game",
            target=34.0,
            injury_status=0.1,
            weather=-5.0,
            pace=61.0,
            weight=0.9,
            game_id="2023-DEN-BUF",
        ),
    ]
    forecaster = PlayerPropForecaster()
    forecaster.fit_pipelines(rows)
    projection = forecaster.probability(
        "Courtland Sutton",
        "receiving_yards",
        "over",
        75.0,
        "game",
        {"opponent": "BUF", "injury_status": 0.2, "weather": -5.0, "pace": 61.0},
    )
    assert 0.0 <= projection.win <= 1.0
    covariance = forecaster._component_covariance(
        "Courtland Sutton",
        "J.K. Dobbins",
        "receiving_yards",
        "game",
        {"opponent": "BUF", "injury_status": 0.2, "weather": -5.0, "pace": 61.0},
        {"opponent": "BUF", "injury_status": 0.1, "weather": -5.0, "pace": 61.0},
    )
    assert abs(covariance) > 1e-6


def test_either_probability_uses_covariance() -> None:
    pytest.importorskip("numpy")
    model = PlayerPropForecaster(
        [
            PlayerProjection("Player A", "receiving_yards", 78.0, 12.0),
            PlayerProjection("Player B", "receiving_yards", 68.0, 11.0),
        ]
    )
    model.register_covariance(
        "Player A",
        "receiving_yards",
        "game",
        "Player B",
        "receiving_yards",
        "game",
        -60.0,
    )
    quote = OddsQuote(
        event_id="2024-NE-NYJ",
        sportsbook="testbook",
        book_market_group="Either Player",
        market="receiving_yards",
        scope="game",
        entity_type="either",
        team_or_player="Player A/Player B",
        side="yes",
        line=75.0,
        american_odds=-110,
        observed_at=dt.datetime.now(dt.timezone.utc),
        extra={"participants": ["Player A", "Player B"]},
    )
    detector = EdgeDetector(value_threshold=-1.0, player_model=model)
    probability = detector._probability_for_quote(quote, None)
    assert probability is not None
    single_a = model.probability("Player A", "receiving_yards", "over", 75.0, "game", {})
    single_b = model.probability("Player B", "receiving_yards", "over", 75.0, "game", {})
    independence = 1.0 - (1.0 - single_a.win) * (1.0 - single_b.win)
    assert abs(probability.win - independence) > 1e-3

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


def test_service_metrics_and_validation(
    tmp_db_path: Path,
    scraper_configs: List[Dict[str, Any]],
    alert_sink: "RecordingAlertSink",
) -> None:
    service = OddsIngestionService(
        scrapers=None,
        scraper_configs=scraper_configs,
        storage_path=tmp_db_path,
        stale_after=dt.timedelta(days=1800),
        alert_sink=alert_sink,
    )
    stored = asyncio.run(service.fetch_and_store())
    metrics = service.metrics
    assert metrics["requested"] == 20
    assert metrics["persisted"] == len(stored) == 17
    assert metrics["discarded"].get("invalid_odds") == 1
    assert metrics["discarded"].get("stale") == 2
    assert metrics["latency_seconds"] >= 0.0
    per_scraper = metrics["per_scraper"]
    assert set(per_scraper) == {"fanduel", "draftkings", "pinnacle"}
    assert all("count" in info for info in per_scraper.values())
    assert all("latency_seconds" in info for info in per_scraper.values())
    assert all(info.get("error") is None for info in per_scraper.values())
    latest = service.load_latest("2024-NE-NYJ")
    assert any(row.sportsbook == "pinnacle" for row in latest)
    assert all(row.american_odds != 0 for row in latest)
    assert alert_sink.messages
    validation_subjects = {subject for subject, _, _ in alert_sink.messages}
    assert "Odds validation issues" in validation_subjects


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
