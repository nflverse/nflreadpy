import asyncio
import datetime as dt

from nflreadpy.betting import (
    LineMovementAnalyzer,
    NameNormalizer,
    PortfolioManager,
    QuantumPortfolioOptimizer,
)
from nflreadpy.betting.analytics import Opportunity
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.scrapers.base import MultiScraperCoordinator, OddsQuote, StaticScraper


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

