import asyncio
import datetime as dt
import random
import statistics

import pytest

from nflreadpy.betting import (
    LineMovementAnalyzer,
    NameNormalizer,
    PortfolioManager,
    QuantumPortfolioOptimizer,
)
from nflreadpy.betting.analytics import KellyCriterion, Opportunity, PortfolioPosition
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.scrapers.base import MultiScraperCoordinator, OddsQuote, StaticScraper
from nflreadpy.betting.utils import (
    american_to_decimal,
    american_to_fractional,
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
)


def test_name_normalizer_applies_aliases() -> None:
    normalizer = NameNormalizer()
    quote = OddsQuote(
        event_id="E1",
        sportsbook="Draft Kings",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="patriots",
        side=None,
        line=None,
        american_odds=-110,
    )
    normalised = normalizer.normalise_quote(quote)
    assert normalised.team_or_player == "NE"
    assert normalised.sportsbook == "draft_kings"


def test_multi_scraper_coordinator_normalises_quotes() -> None:
    quote = OddsQuote(
        event_id="E1",
        sportsbook="Fan Duel",
        book_market_group="Game Lines",
        market="spread",
        scope="game",
        entity_type="team",
        team_or_player="jets",
        side="dog",
        line=3.5,
        american_odds=+115,
    )
    scraper = StaticScraper("fanduel", [quote])
    coordinator = MultiScraperCoordinator([scraper])
    result = asyncio.run(coordinator.collect_once())
    assert result[0].team_or_player == "NYJ"


def test_portfolio_manager_limits_event_exposure() -> None:
    manager = PortfolioManager(bankroll=100.0, max_risk_per_bet=0.1, max_event_exposure=0.1)
    opportunity = Opportunity(
        event_id="E1",
        sportsbook="book",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=+120,
        model_probability=0.55,
        push_probability=0.0,
        implied_probability=0.45,
        expected_value=0.08,
        kelly_fraction=0.2,
        extra={},
    )
    first = manager.allocate(opportunity)
    assert first is not None
    second = manager.allocate(opportunity)
    assert second is None


def test_odds_conversion_helpers_roundtrip() -> None:
    decimal = american_to_decimal(+150)
    assert decimal == pytest.approx(2.5)
    assert decimal_to_american(decimal) == 150
    decimal_fraction = decimal_to_fractional(decimal)
    assert decimal_fraction == (3, 2)
    frac = american_to_fractional(-120)
    assert fractional_to_decimal(*frac) == pytest.approx(1.0 + 5 / 6)
    assert fractional_to_american(*frac) == -120
    assert implied_probability_from_decimal(decimal) == pytest.approx(0.4)
    assert implied_probability_from_fractional(*frac) == pytest.approx(0.54545, rel=1e-4)
    assert implied_probability_from_american(+150) == pytest.approx(0.4)


def test_implied_probability_bidirectional_helpers() -> None:
    probability = 0.4
    decimal = implied_probability_to_decimal(probability)
    assert decimal == pytest.approx(2.5)
    american = implied_probability_to_american(probability)
    assert american == 150
    fractional = implied_probability_to_fraction(probability)
    assert fractional == (3, 2)
    assert implied_probability_from_decimal(decimal) == pytest.approx(probability)
    assert implied_probability_from_american(american) == pytest.approx(probability)
    assert implied_probability_from_fractional(*fractional) == pytest.approx(probability)


def test_kelly_fraction_respects_fractional_multiplier() -> None:
    base = KellyCriterion.fraction(0.55, 0.45, +100)
    half = KellyCriterion.fraction(0.55, 0.45, +100, fractional_kelly=0.5)
    assert half == pytest.approx(base * 0.5)
    capped = KellyCriterion.fraction(0.55, 0.45, +100, cap=0.05)
    assert capped <= 0.05


def test_kelly_fraction_supports_multiple_odds_formats() -> None:
    american = KellyCriterion.fraction(0.55, 0.45, +120)
    decimal = KellyCriterion.fraction_from_decimal(0.55, 0.45, 2.2)
    fractional = KellyCriterion.fraction_from_fractional(0.55, 0.45, 6, 5)
    assert decimal == pytest.approx(american)
    assert fractional == pytest.approx(american)
    scaled = KellyCriterion.fraction_from_decimal(
        0.55, 0.45, 2.2, fractional_kelly=0.5
    )
    assert scaled == pytest.approx(decimal * 0.5)


def test_kelly_fraction_from_implied_probability_matches_decimal() -> None:
    implied = 0.4
    decimal_fraction = KellyCriterion.fraction_from_decimal(0.55, 0.45, 2.5)
    implied_fraction = KellyCriterion.fraction_from_implied_probability(
        0.55, 0.45, implied
    )
    assert implied_fraction == pytest.approx(decimal_fraction)


def test_kelly_instance_applies_configured_defaults() -> None:
    criterion = KellyCriterion(fractional_kelly=0.4, cap=0.15)
    via_callable = criterion(0.6, 0.4, +110)
    expected = KellyCriterion.fraction(0.6, 0.4, +110, fractional_kelly=0.4, cap=0.15)
    assert via_callable == pytest.approx(expected)
    assert criterion.last_fraction == pytest.approx(via_callable)
    updated = criterion.with_fractional_kelly(0.25)
    assert updated.fractional_kelly == pytest.approx(0.25)
    capped = updated.with_cap(0.05)
    assert capped.cap == pytest.approx(0.05)


def test_portfolio_manager_correlation_and_simulation() -> None:
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=1.0,
        max_event_exposure=1.0,
        fractional_kelly=1.0,
        correlation_limits={"team": 0.5},
    )
    winning = Opportunity(
        event_id="E1",
        sportsbook="book",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=+100,
        model_probability=1.0,
        push_probability=0.0,
        implied_probability=0.5,
        expected_value=0.5,
        kelly_fraction=1.0,
        extra={"correlation_group": "team"},
    )
    first = manager.allocate(winning)
    assert first is not None
    second = manager.allocate(winning)
    assert second is None
    simulation = manager.simulate_bankroll(trials=3, seed=1)
    assert simulation.trials == 3
    assert simulation.terminal_balances[0] == pytest.approx(150.0)
    summary = simulation.summary()
    assert summary["average_drawdown"] == pytest.approx(0.0)
    assert summary["trials"] == pytest.approx(3.0)
    assert summary["worst_terminal"] <= summary["mean_terminal"]
    assert summary["p95_drawdown"] >= summary["p05_drawdown"]


def test_portfolio_manager_setters_update_limits() -> None:
    manager = PortfolioManager(bankroll=100.0, max_risk_per_bet=0.1, max_event_exposure=0.2)
    manager.set_fractional_kelly(0.3)
    assert manager.fractional_kelly == pytest.approx(0.3)
    manager.set_correlation_limit("same_game", 0.4)
    assert manager.correlation_limits["same_game"] == pytest.approx(0.4)
    manager.set_correlation_limit("same_game", None)
    assert "same_game" not in manager.correlation_limits


def test_portfolio_manager_tracks_last_simulation() -> None:
    manager = PortfolioManager(bankroll=200.0, max_risk_per_bet=1.0, max_event_exposure=1.0)
    opportunity = Opportunity(
        event_id="E1",
        sportsbook="book",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=+120,
        model_probability=0.55,
        push_probability=0.05,
        implied_probability=0.45,
        expected_value=0.1,
        kelly_fraction=0.15,
        extra={},
    )
    position = PortfolioPosition(opportunity=opportunity, stake=20.0)
    result = manager.simulate_bankroll(trials=5, seed=4, positions=[position])
    assert manager.last_simulation is result
    summary = manager.bankroll_summary()
    assert summary is not None
    assert summary["trials"] == pytest.approx(5.0)
    assert summary == result.summary()


def test_odds_quote_helpers_expose_conversions() -> None:
    quote = OddsQuote(
        event_id="E1",
        sportsbook="book",
        book_market_group="Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=-120,
    )
    assert quote.decimal_odds() == pytest.approx(1.8333333, rel=1e-6)
    assert quote.fractional_odds() == (5, 6)
    assert quote.profit_multiplier() == pytest.approx(0.8333333, rel=1e-6)
    assert quote.decimal_multiplier() == pytest.approx(quote.profit_multiplier())
    assert quote.implied_probability() == pytest.approx(0.5454545, rel=1e-6)


def test_bankroll_simulation_is_deterministic() -> None:
    manager = PortfolioManager(bankroll=100.0, max_risk_per_bet=1.0, max_event_exposure=1.0)
    opportunity = Opportunity(
        event_id="E1",
        sportsbook="book",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=+110,
        model_probability=0.6,
        push_probability=0.1,
        implied_probability=0.476,
        expected_value=0.08,
        kelly_fraction=0.2,
        extra={},
    )
    position = PortfolioPosition(opportunity=opportunity, stake=10.0)
    simulation = manager.simulate_bankroll(trials=4, seed=7, positions=[position])
    rng = random.Random(7)
    expected_balances: list[list[float]] = []
    start = 100.0
    payout = opportunity.decimal_multiplier()
    for _ in range(4):
        bankroll = start
        balances = [bankroll]
        roll = rng.random()
        if roll < opportunity.model_probability:
            bankroll += position.stake * payout
        elif roll < opportunity.model_probability + opportunity.push_probability:
            bankroll += 0.0
        else:
            bankroll -= position.stake
        balances.append(bankroll)
        expected_balances.append(balances)
    assert [trajectory.balances for trajectory in simulation.trajectories] == expected_balances
    summary = simulation.summary()
    terminals = [balances[-1] for balances in expected_balances]
    drawdowns = [_max_drawdown(balances) for balances in expected_balances]
    assert summary["mean_terminal"] == pytest.approx(statistics.mean(terminals))
    assert summary["median_terminal"] == pytest.approx(statistics.median(terminals))
    assert summary["worst_terminal"] == pytest.approx(min(terminals))
    assert summary["average_drawdown"] == pytest.approx(statistics.mean(drawdowns))
    assert summary["worst_drawdown"] == pytest.approx(max(drawdowns))


def _max_drawdown(values: list[float]) -> float:
    peak = -float("inf")
    max_dd = 0.0
    for value in values:
        peak = value if peak == -float("inf") else max(peak, value)
        if peak <= 0:
            continue
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd


def test_line_movement_analyzer_orders_by_magnitude() -> None:
    timestamp = dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc)
    history = [
        IngestedOdds(
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
            observed_at=timestamp,
            extra={},
        ),
        IngestedOdds(
            event_id="E1",
            sportsbook="book",
            book_market_group="Game Lines",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side="fav",
            line=-3.5,
            american_odds=-125,
            observed_at=timestamp + dt.timedelta(minutes=10),
            extra={},
        ),
    ]
    analyzer = LineMovementAnalyzer(history)
    movements = analyzer.summarise()
    assert movements[0].delta == -15


def test_quantum_optimizer_prioritises_positive_edges() -> None:
    opportunities = [
        Opportunity(
            event_id=f"E{i}",
            sportsbook="book",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="T1",
            side=None,
            line=None,
            american_odds=+120,
            model_probability=0.55 + i * 0.05,
            push_probability=0.0,
            implied_probability=0.45,
            expected_value=0.05 + i * 0.02,
            kelly_fraction=0.1 + i * 0.05,
            extra={},
        )
        for i in range(3)
    ]
    optimizer = QuantumPortfolioOptimizer(shots=256, seed=1)
    ranking = optimizer.optimise(opportunities)
    assert ranking[0][0].expected_value >= ranking[-1][0].expected_value

