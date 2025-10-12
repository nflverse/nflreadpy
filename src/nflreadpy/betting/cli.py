"""Command line entry point for the Bloomberg-style betting stack."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Dict, Iterable, List, Sequence

from . import (
    EdgeDetector,
    GameSimulationConfig,
    LineMovementAnalyzer,
    MockSportsbookScraper,
    MonteCarloEngine,
    OddsIngestionService,
    Opportunity,
    PlayerPropForecaster,
    PortfolioManager,
    QuantumPortfolioOptimizer,
    consolidate_best_prices,
)
from .models import PlayerProjection, TeamRating


def _build_engine(teams: Iterable[str], iterations: int) -> MonteCarloEngine:
    ratings: Dict[str, TeamRating] = {}
    for index, team in enumerate(sorted(set(teams))):
        ratings[team] = TeamRating(
            team=team,
            offensive_rating=0.8 + 0.1 * (index % 4),
            defensive_rating=0.5 + 0.05 * (index % 5),
        )
    return MonteCarloEngine(ratings, GameSimulationConfig(iterations=iterations, seed=7))


def _build_player_model() -> PlayerPropForecaster:
    projections = [
        PlayerProjection(
            player="Patrick Mahomes",
            market="passing_rushing_yards",
            mean=340.0,
            stdev=28.0,
        ),
        PlayerProjection(
            player="Courtland Sutton",
            market="two_plus_touchdowns",
            mean=0.32,
            stdev=0.18,
            distribution="poisson",
        ),
    ]
    return PlayerPropForecaster(projections)


async def _collect_quotes(service: OddsIngestionService) -> List[Opportunity]:
    stored = await service.fetch_and_store()
    teams: Dict[str, set[str]] = {}
    for quote in stored:
        if quote.entity_type == "team" and quote.market == "moneyline":
            teams.setdefault(quote.event_id, set()).add(quote.team_or_player)
    engine = _build_engine({team for grouping in teams.values() for team in grouping}, 20_000)
    simulations = []
    for event_id, participants in teams.items():
        if len(participants) != 2:
            continue
        home, away = sorted(participants)
        simulations.append(engine.simulate_game(event_id, home, away))
    detector = EdgeDetector(
        value_threshold=0.01,
        player_model=_build_player_model(),
    )
    opportunities = detector.detect(stored, simulations)
    return consolidate_best_prices(opportunities)


def _render_opportunities(opportunities: Sequence[Opportunity]) -> None:
    if not opportunities:
        print("No actionable opportunities identified.")
        return
    header = (
        f"{'Event':<16} {'Market':<20} {'Selection':<22} {'Odds':>6}"
        f" {'Model P':>8} {'EV':>8} {'Kelly':>8}"
    )
    print(header)
    print("-" * len(header))
    for opp in opportunities:
        print(
            f"{opp.event_id:<16} {opp.market:<20} {opp.team_or_player:<22}"
            f" {opp.american_odds:>+6d} {opp.model_probability:>8.3f}"
            f" {opp.expected_value:>8.3f} {opp.kelly_fraction:>8.3f}"
        )


def _portfolio_allocation(opportunities: Sequence[Opportunity], bankroll: float) -> None:
    manager = PortfolioManager(bankroll=bankroll)
    optimizer = QuantumPortfolioOptimizer(shots=256, seed=13)
    optimised = optimizer.optimise(opportunities)
    print("\nQuantum-inspired ranking (top states):")
    for opp, weight in optimised[:5]:
        print(f"  {opp.event_id} {opp.market} {opp.team_or_player}: {weight:.2%}")
    print("\nPortfolio allocations:")
    for opp in opportunities:
        position = manager.allocate(opp)
        if not position:
            continue
        print(
            f"  Stake {position.stake:>6.2f} units on {opp.event_id}"
            f" {opp.market} {opp.team_or_player} @ {opp.american_odds:+d}"
        )
    print("\nExposure by event:")
    print(json.dumps(manager.exposure_report(), indent=2, default=str))


def _line_movement(service: OddsIngestionService, limit: int) -> None:
    history = service.load_history(limit=limit)
    analyzer = LineMovementAnalyzer(history)
    movements = analyzer.summarise()[:5]
    if not movements:
        print("No line movement recorded yet.")
        return
    print("\nTop line movements:")
    for movement in movements:
        key = movement.key
        print(
            f"  {key[0]} {key[1]} {key[3]} {key[4]} {key[5]}"
            f" {movement.opening_price:+d}->{movement.latest_price:+d}"
        )


async def _async_main(args: argparse.Namespace) -> None:
    scraper = MockSportsbookScraper()
    service = OddsIngestionService([scraper], storage_path=args.storage)
    opportunities = await _collect_quotes(service)
    _render_opportunities(opportunities)
    _portfolio_allocation(opportunities, bankroll=args.bankroll)
    _line_movement(service, limit=args.history_limit)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="betting_cli.sqlite3")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--history-limit", type=int, default=128)
    args = parser.parse_args(argv)
    asyncio.run(_async_main(args))


__all__ = ["main"]

