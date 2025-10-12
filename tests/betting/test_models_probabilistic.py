"""Regression tests for probabilistic betting models."""

from __future__ import annotations

import datetime as dt

import pytest

from nflreadpy.betting.analytics import EdgeDetector
from nflreadpy.betting.models import (
    BivariatePoissonEngine,
    GameSimulationConfig,
    HistoricalGameRecord,
    PlayerFeatureRow,
    PlayerPropForecaster,
    SimulationResult,
    TeamRating,
)
from nflreadpy.betting.scrapers.base import OddsQuote


@pytest.fixture()
def _sample_ratings() -> dict[str, TeamRating]:
    return {
        "NE": TeamRating(team="NE", offensive_rating=1.2, defensive_rating=-0.3),
        "BUF": TeamRating(team="BUF", offensive_rating=0.9, defensive_rating=-0.1),
        "KC": TeamRating(team="KC", offensive_rating=1.6, defensive_rating=-0.4),
    }


@pytest.fixture()
def _historical_games(_sample_ratings: dict[str, TeamRating]) -> list[HistoricalGameRecord]:
    ratings = _sample_ratings
    return [
        HistoricalGameRecord(
            home_team="NE",
            away_team="BUF",
            home_points=27,
            away_points=21,
            home_pace=64.2,
            away_pace=62.8,
            home_offense_rating=ratings["NE"].offensive_rating,
            home_defense_rating=ratings["NE"].defensive_rating,
            away_offense_rating=ratings["BUF"].offensive_rating,
            away_defense_rating=ratings["BUF"].defensive_rating,
        ),
        HistoricalGameRecord(
            home_team="KC",
            away_team="BUF",
            home_points=31,
            away_points=24,
            home_pace=66.5,
            away_pace=63.3,
            home_offense_rating=ratings["KC"].offensive_rating,
            home_defense_rating=ratings["KC"].defensive_rating,
            away_offense_rating=ratings["BUF"].offensive_rating,
            away_defense_rating=ratings["BUF"].defensive_rating,
        ),
        HistoricalGameRecord(
            home_team="BUF",
            away_team="NE",
            home_points=24,
            away_points=20,
            home_pace=62.0,
            away_pace=61.2,
            home_offense_rating=ratings["BUF"].offensive_rating,
            home_defense_rating=ratings["BUF"].defensive_rating,
            away_offense_rating=ratings["NE"].offensive_rating,
            away_defense_rating=ratings["NE"].defensive_rating,
        ),
        HistoricalGameRecord(
            home_team="KC",
            away_team="NE",
            home_points=30,
            away_points=23,
            home_pace=67.2,
            away_pace=63.7,
            home_offense_rating=ratings["KC"].offensive_rating,
            home_defense_rating=ratings["KC"].defensive_rating,
            away_offense_rating=ratings["NE"].offensive_rating,
            away_defense_rating=ratings["NE"].defensive_rating,
        ),
    ]


def test_bivariate_poisson_engine_generates_correlated_distribution(
    _sample_ratings: dict[str, TeamRating],
    _historical_games: list[HistoricalGameRecord],
) -> None:
    engine = BivariatePoissonEngine(
        _sample_ratings,
        config=GameSimulationConfig(iterations=2048),
        historical_games=_historical_games,
        backend="python",
    )
    result = engine.simulate_game("2024-NE-BUF", "NE", "BUF")

    assert pytest.approx(sum(result.total_distribution.values()), rel=1e-6) == 1.0
    assert result.home_rate > 0
    assert result.away_rate > 0
    assert -1.0 <= result.correlation(("home_score", ""), ("away_score", "")) <= 1.0
    assert pytest.approx(result.expected_total, rel=1e-3) == pytest.approx(
        result.home_mean + result.away_mean, rel=1e-3
    )

    benchmark = engine.benchmark([("g1", "NE", "BUF")], repeats=1)
    assert benchmark.simulations_run >= 1
    assert benchmark.elapsed_seconds >= 0.0


def test_player_prop_forecaster_pipelines_and_covariance() -> None:
    rows = [
        PlayerFeatureRow(
            player="A. Receiver",
            opponent="NYJ",
            market="receptions",
            scope="game",
            target=7.4,
            injury_status=0.1,
            weather=-0.3,
            pace=65.0,
            usage=0.23,
            travel=1.2,
            weight=1.1,
            game_id="game-1",
        ),
        PlayerFeatureRow(
            player="B. Receiver",
            opponent="NYJ",
            market="receptions",
            scope="game",
            target=6.2,
            injury_status=0.05,
            weather=-0.3,
            pace=65.0,
            usage=0.19,
            travel=1.1,
            weight=1.0,
            game_id="game-1",
        ),
        PlayerFeatureRow(
            player="A. Receiver",
            opponent="BUF",
            market="receptions",
            scope="game",
            target=8.6,
            injury_status=0.2,
            weather=0.15,
            pace=66.0,
            usage=0.27,
            travel=0.8,
            weight=1.0,
            game_id="game-2",
        ),
        PlayerFeatureRow(
            player="B. Receiver",
            opponent="BUF",
            market="receptions",
            scope="game",
            target=5.8,
            injury_status=0.08,
            weather=0.15,
            pace=66.0,
            usage=0.22,
            travel=0.7,
            weight=1.0,
            game_id="game-2",
        ),
    ]

    forecaster = PlayerPropForecaster()
    forecaster.fit_pipelines(rows, markets=["receptions"], distribution="normal")

    features_a = {
        "opponent": "NYJ",
        "injury_status": 0.1,
        "weather": -0.2,
        "pace": 65.5,
        "usage": 0.25,
        "travel": 1.0,
    }
    projection = forecaster.probability(
        "A. Receiver",
        "receptions",
        "over",
        6.5,
        "game",
        features_a,
    )
    assert 0.0 <= projection.win <= 1.0

    features_b = {
        "opponent": "NYJ",
        "injury_status": 0.05,
        "weather": -0.1,
        "pace": 65.0,
        "usage": 0.2,
        "travel": 1.2,
    }
    correlation = forecaster.component_correlation(
        "A. Receiver",
        "B. Receiver",
        "receptions",
        "game",
        features_a,
        features_b,
    )
    assert -1.0 <= correlation <= 1.0

    composite = forecaster.probability(
        "Combo",
        "receptions",
        "over",
        13.0,
        "game",
        {
            "components": ["A. Receiver", "B. Receiver"],
            "component_features": {"A. Receiver": features_a, "B. Receiver": features_b},
        },
    )
    assert 0.0 <= composite.win <= 1.0


def test_edge_detector_applies_correlation_penalty() -> None:
    result = SimulationResult(
        event_id="sim-1",
        home_team="NE",
        away_team="BUF",
        iterations=1000,
        home_win_probability=1.0,
        away_win_probability=0.0,
        expected_margin=4.0,
        expected_total=46.4,
        margin_distribution={4: 1.0},
        total_distribution={44: 0.6, 50: 0.4},
        home_score_distribution={24: 0.6, 27: 0.4},
        away_score_distribution={20: 0.6, 23: 0.4},
        home_rate=25.0,
        away_rate=21.0,
        shared_rate=3.0,
        home_mean=25.2,
        away_mean=21.2,
        home_variance=2.16,
        away_variance=2.16,
        correlation_matrix={
            ("home_score", "away_score"): 0.6,
            ("away_score", "home_score"): 0.6,
            ("home_score", "total"): 0.8,
            ("total", "home_score"): 0.8,
            ("away_score", "total"): 0.65,
            ("total", "away_score"): 0.65,
            ("total", "total"): 1.0,
        },
    )

    quote = OddsQuote(
        event_id="sim-1",
        sportsbook="testbook",
        book_market_group="totals",
        market="game_total",
        scope="game",
        entity_type="total",
        team_or_player="game",
        side="over",
        line=45.5,
        american_odds=-110,
        observed_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        extra={},
    )

    detector_neutral = EdgeDetector(value_threshold=0.0, correlation_penalty=0.0)
    detector_correlated = EdgeDetector(value_threshold=0.0, correlation_penalty=0.5)

    neutral_edge = detector_neutral.detect([quote], [result])[0]
    correlated_edge = detector_correlated.detect([quote], [result])[0]

    assert neutral_edge.expected_value < 0.0
    assert correlated_edge.expected_value > neutral_edge.expected_value
    assert correlated_edge.expected_value > 0.0
