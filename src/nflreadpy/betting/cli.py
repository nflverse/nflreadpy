"""Command line interface for the Bloomberg-style betting stack."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from . import (
    Dashboard,
    EdgeDetector,
    GameSimulationConfig,
    LineMovementAnalyzer,
    MockSportsbookScraper,
    MonteCarloEngine,
    OddsIngestionService,
    Opportunity,
    PlayerPropForecaster,
    PortfolioManager,
    BankrollSimulationResult,
    QuantumPortfolioOptimizer,
    consolidate_best_prices,
)
from .alerts import AlertManager, get_alert_manager
from .analytics import LineMovement
from .dashboard import RiskSummary
from .ingestion import IngestedOdds
from .models import PlayerProjection, SimulationResult, TeamRating
from .scheduler import Scheduler
from .scrapers.base import OddsQuote


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


def _quotes_from_ingested(records: Sequence[IngestedOdds]) -> List[OddsQuote]:
    return [
        OddsQuote(
            event_id=record.event_id,
            sportsbook=record.sportsbook,
            book_market_group=record.book_market_group,
            market=record.market,
            scope=record.scope,
            entity_type=record.entity_type,
            team_or_player=record.team_or_player,
            side=record.side,
            line=record.line,
            american_odds=record.american_odds,
            observed_at=record.observed_at,
            extra=record.extra,
        )
        for record in records
    ]


def _run_simulations(quotes: Sequence[OddsQuote], iterations: int) -> List[SimulationResult]:
    teams: Dict[str, set[str]] = defaultdict(set)
    for quote in quotes:
        if quote.entity_type == "team" and quote.market == "moneyline":
            teams[quote.event_id].add(quote.team_or_player)
    participants = {team for grouping in teams.values() for team in grouping}
    if not participants:
        return []
    engine = _build_engine(participants, iterations)
    simulations: List[SimulationResult] = []
    for event_id, entries in teams.items():
        if len(entries) != 2:
            continue
        home, away = sorted(entries)
        simulations.append(engine.simulate_game(event_id, home, away))
    return simulations


def _parse_correlation_limits(values: Sequence[str] | None) -> Dict[str, float]:
    limits: Dict[str, float] = {}
    if not values:
        return limits
    for entry in values:
        if not entry:
            continue
        separator = "=" if "=" in entry else ":" if ":" in entry else None
        if separator is None:
            raise ValueError(
                "Correlation limits must be provided as 'group=value' or 'group:value'"
            )
        group, raw_value = entry.split(separator, 1)
        group = group.strip()
        if not group:
            raise ValueError("Correlation group cannot be empty")
        try:
            limit = float(raw_value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid correlation limit '{entry}'") from exc
        limits[group] = limit
    return limits


def _build_portfolio(
    opportunities: Sequence[Opportunity],
    *,
    bankroll: float,
    portfolio_fraction: float,
    correlation_limits: Mapping[str, float],
    risk_trials: int,
    risk_seed: int | None,
) -> Tuple[PortfolioManager, BankrollSimulationResult | None]:
    manager = PortfolioManager(
        bankroll=bankroll,
        fractional_kelly=portfolio_fraction,
        correlation_limits=correlation_limits,
    )
    for opportunity in opportunities:
        manager.allocate(opportunity)
    simulation: BankrollSimulationResult | None = None
    if risk_trials > 0 and manager.positions:
        simulation = manager.simulate_bankroll(trials=risk_trials, seed=risk_seed)
    return manager, simulation


def _detect_opportunities(
    quotes: Sequence[OddsQuote],
    simulations: Sequence[SimulationResult],
    *,
    value_threshold: float,
    alert_manager: AlertManager | None,
    kelly_fraction: float,
) -> List[Opportunity]:
    detector = EdgeDetector(
        value_threshold=value_threshold,
        player_model=_build_player_model(),
        alert_manager=alert_manager,
        kelly_fraction=kelly_fraction,
    )
    opportunities = detector.detect(quotes, simulations)
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


def _portfolio_allocation(
    opportunities: Sequence[Opportunity],
    *,
    bankroll: float,
    portfolio_fraction: float,
    correlation_limits: Mapping[str, float],
    risk_trials: int,
    risk_seed: int | None,
) -> Tuple[PortfolioManager, BankrollSimulationResult | None]:
    manager, simulation = _build_portfolio(
        opportunities,
        bankroll=bankroll,
        portfolio_fraction=portfolio_fraction,
        correlation_limits=correlation_limits,
        risk_trials=risk_trials,
        risk_seed=risk_seed,
    )
    optimizer = QuantumPortfolioOptimizer(shots=256, seed=13)
    optimised = optimizer.optimise(opportunities)
    print("\nQuantum-inspired ranking (top states):")
    for opp, weight in optimised[:5]:
        print(f"  {opp.event_id} {opp.market} {opp.team_or_player}: {weight:.2%}")
    print("\nPortfolio allocations:")
    for position in manager.positions:
        opp = position.opportunity
        print(
            f"  Stake {position.stake:>6.2f} units on {opp.event_id}"
            f" {opp.market} {opp.team_or_player} @ {opp.american_odds:+d}"
        )
    print("\nExposure by event:")
    print(json.dumps(manager.exposure_report(), indent=2, default=str))
    correlation = manager.correlation_report()
    if correlation:
        print("\nCorrelation exposure:")
        print(json.dumps(correlation, indent=2, default=str))
    if simulation:
        summary = simulation.summary()
        print("\nBankroll simulation summary:")
        print(
            json.dumps(
                {
                    "trials": int(summary["trials"]),
                    "mean_terminal": summary["mean_terminal"],
                    "median_terminal": summary["median_terminal"],
                    "worst_terminal": summary["worst_terminal"],
                    "average_drawdown": summary["average_drawdown"],
                    "worst_drawdown": summary["worst_drawdown"],
                    "p05_drawdown": summary["p05_drawdown"],
                    "p95_drawdown": summary["p95_drawdown"],
                },
                indent=2,
            )
        )
    return manager, simulation


def _line_movement(
    service: OddsIngestionService,
    *,
    limit: int,
    alert_manager: AlertManager | None,
    threshold: int,
) -> List[LineMovement]:
    history = service.load_history(limit=limit)
    analyzer = LineMovementAnalyzer(
        history,
        alert_manager=alert_manager,
        alert_threshold=threshold,
    )
    return analyzer.summarise()


def _print_line_movements(movements: Sequence[LineMovement]) -> None:
    if not movements:
        print("No line movement recorded yet.")
        return
    print("\nTop line movements:")
    for movement in movements[:5]:
        key = movement.key
        print(
            f"  {key[0]} {key[1]} {key[3]} {key[4]} {key[5]}"
            f" {movement.opening_price:+d}->{movement.latest_price:+d}"
        )


async def _cmd_ingest(args: argparse.Namespace, alert_manager: AlertManager | None) -> None:
    service = OddsIngestionService([MockSportsbookScraper()], storage_path=args.storage)
    if args.interval <= 0:
        stored = await service.fetch_and_store()
        print(f"Ingested {len(stored)} quotes into {args.storage}")
        return

    scheduler = Scheduler()

    async def _collect() -> None:
        await service.fetch_and_store()

    scheduler.add_job(
        _collect,
        interval=args.interval,
        jitter=args.jitter,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        name="odds-ingest",
    )
    print("Starting ingestion scheduler. Press Ctrl+C to stop.")
    try:
        await scheduler.run()
    except KeyboardInterrupt:
        scheduler.stop()
    finally:
        await scheduler.shutdown()


async def _cmd_simulate(args: argparse.Namespace, alert_manager: AlertManager | None) -> None:
    service = OddsIngestionService([MockSportsbookScraper()], storage_path=args.storage)
    if args.refresh:
        ingested = await service.fetch_and_store()
    else:
        ingested = service.load_latest()
        if not ingested:
            ingested = await service.fetch_and_store()
    quotes = _quotes_from_ingested(ingested)
    simulations = _run_simulations(quotes, args.iterations)
    correlation_limits = _parse_correlation_limits(args.correlation_limit)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
        kelly_fraction=args.kelly_fraction,
    )
    _render_opportunities(opportunities)
    manager, simulation = _portfolio_allocation(
        opportunities,
        bankroll=args.bankroll,
        portfolio_fraction=args.portfolio_fraction,
        correlation_limits=correlation_limits,
        risk_trials=args.risk_trials,
        risk_seed=args.risk_seed,
    )
    movements = _line_movement(
        service,
        limit=args.history_limit,
        alert_manager=alert_manager,
        threshold=args.movement_threshold,
    )
    _print_line_movements(movements)


async def _cmd_scan(args: argparse.Namespace, alert_manager: AlertManager | None) -> None:
    service = OddsIngestionService([MockSportsbookScraper()], storage_path=args.storage)
    history = service.load_history(limit=args.history_limit)
    if not history:
        history = await service.fetch_and_store()
    quotes = _quotes_from_ingested(history)
    simulations = _run_simulations(quotes, args.iterations)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
        kelly_fraction=args.kelly_fraction,
    )
    _render_opportunities(opportunities)
    movements = _line_movement(
        service,
        limit=args.history_limit,
        alert_manager=alert_manager,
        threshold=args.movement_threshold,
    )
    _print_line_movements(movements)


async def _cmd_dashboard(args: argparse.Namespace, alert_manager: AlertManager | None) -> None:
    service = OddsIngestionService([MockSportsbookScraper()], storage_path=args.storage)
    if args.refresh:
        latest = await service.fetch_and_store()
    else:
        latest = service.load_latest()
        if not latest:
            latest = await service.fetch_and_store()
    quotes = _quotes_from_ingested(latest)
    simulations = _run_simulations(quotes, args.iterations)
    correlation_limits = _parse_correlation_limits(args.correlation_limit)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
        kelly_fraction=args.kelly_fraction,
    )
    manager, simulation = _build_portfolio(
        opportunities,
        bankroll=args.bankroll,
        portfolio_fraction=args.portfolio_fraction,
        correlation_limits=correlation_limits,
        risk_trials=args.risk_trials,
        risk_seed=args.risk_seed,
    )
    dashboard = Dashboard()
    risk_summary = RiskSummary(
        bankroll=args.bankroll,
        opportunity_fraction=args.kelly_fraction,
        portfolio_fraction=args.portfolio_fraction,
        positions=tuple(manager.positions),
        exposure_by_event=manager.exposure_report(),
        correlation_exposure=manager.correlation_report(),
        simulation=simulation,
    )
    rendered = dashboard.render(latest, simulations, opportunities, risk_summary=risk_summary)
    print(rendered)


async def _cmd_backtest(args: argparse.Namespace, alert_manager: AlertManager | None) -> None:
    service = OddsIngestionService([MockSportsbookScraper()], storage_path=args.storage)
    history = service.load_history(limit=args.limit)
    if not history:
        print("No historical odds available for backtesting.")
        return
    grouped: Dict[str, List[IngestedOdds]] = defaultdict(list)
    for row in history:
        grouped[row.observed_at.isoformat()].append(row)
    cumulative_ev = 0.0
    sample_count = 0
    for timestamp in sorted(grouped):
        records = grouped[timestamp]
        quotes = _quotes_from_ingested(records)
        simulations = _run_simulations(quotes, args.iterations)
        opportunities = _detect_opportunities(
            quotes,
            simulations,
            value_threshold=args.value_threshold,
            alert_manager=alert_manager,
        )
        cumulative_ev += sum(opp.expected_value for opp in opportunities)
        sample_count += 1
    average_ev = cumulative_ev / sample_count if sample_count else 0.0
    print(
        json.dumps(
            {
                "samples": sample_count,
                "total_expected_value": cumulative_ev,
                "average_expected_value": average_ev,
            },
            indent=2,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--storage", default="betting_cli.sqlite3")
    parent.add_argument("--alerts-config")

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", parents=[parent], help="Collect odds")
    ingest_parser.add_argument("--interval", type=float, default=0.0)
    ingest_parser.add_argument("--jitter", type=float, default=0.0)
    ingest_parser.add_argument("--retries", type=int, default=3)
    ingest_parser.add_argument("--retry-backoff", type=float, default=2.0)
    ingest_parser.set_defaults(handler=_cmd_ingest)

    simulate_parser = subparsers.add_parser(
        "simulate",
        parents=[parent],
        help="Simulate markets and rank opportunities",
    )
    simulate_parser.add_argument("--iterations", type=int, default=20_000)
    simulate_parser.add_argument("--bankroll", type=float, default=1000.0)
    simulate_parser.add_argument("--refresh", action="store_true", default=False)
    simulate_parser.add_argument("--value-threshold", type=float, default=0.01)
    simulate_parser.add_argument("--history-limit", type=int, default=128)
    simulate_parser.add_argument("--movement-threshold", type=int, default=40)
    simulate_parser.add_argument("--kelly-fraction", type=float, default=1.0)
    simulate_parser.add_argument("--portfolio-fraction", type=float, default=1.0)
    simulate_parser.add_argument(
        "--correlation-limit", action="append", default=[], help="Limit correlated exposure"
    )
    simulate_parser.add_argument("--risk-trials", type=int, default=500)
    simulate_parser.add_argument("--risk-seed", type=int)
    simulate_parser.set_defaults(handler=_cmd_simulate)

    scan_parser = subparsers.add_parser(
        "scan",
        parents=[parent],
        help="Scan stored markets for edges and movement",
    )
    scan_parser.add_argument("--iterations", type=int, default=10_000)
    scan_parser.add_argument("--history-limit", type=int, default=256)
    scan_parser.add_argument("--value-threshold", type=float, default=0.02)
    scan_parser.add_argument("--movement-threshold", type=int, default=30)
    scan_parser.add_argument("--kelly-fraction", type=float, default=1.0)
    scan_parser.set_defaults(handler=_cmd_scan)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        parents=[parent],
        help="Render the ASCII dashboard",
    )
    dashboard_parser.add_argument("--iterations", type=int, default=15_000)
    dashboard_parser.add_argument("--refresh", action="store_true", default=False)
    dashboard_parser.add_argument("--value-threshold", type=float, default=0.015)
    dashboard_parser.add_argument("--bankroll", type=float, default=1000.0)
    dashboard_parser.add_argument("--kelly-fraction", type=float, default=1.0)
    dashboard_parser.add_argument("--portfolio-fraction", type=float, default=1.0)
    dashboard_parser.add_argument(
        "--correlation-limit", action="append", default=[], help="Limit correlated exposure"
    )
    dashboard_parser.add_argument("--risk-trials", type=int, default=250)
    dashboard_parser.add_argument("--risk-seed", type=int)
    dashboard_parser.set_defaults(handler=_cmd_dashboard)

    backtest_parser = subparsers.add_parser(
        "backtest",
        parents=[parent],
        help="Replay historical odds and summarise expected value",
    )
    backtest_parser.add_argument("--limit", type=int, default=500)
    backtest_parser.add_argument("--iterations", type=int, default=8_000)
    backtest_parser.add_argument("--value-threshold", type=float, default=0.02)
    backtest_parser.set_defaults(handler=_cmd_backtest)

    return parser


async def _dispatch(args: argparse.Namespace) -> None:
    alert_manager = get_alert_manager(args.alerts_config)
    handler = args.handler
    await handler(args, alert_manager)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_dispatch(args))


__all__ = ["main"]

