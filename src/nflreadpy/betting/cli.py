"""Command line interface for the Bloomberg-style betting stack."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence

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
    QuantumPortfolioOptimizer,
    consolidate_best_prices,
)
from .alerts import AlertManager, get_alert_manager, install_signal_handlers
from .analytics import LineMovement
from .ingestion import IngestedOdds
from .models import PlayerProjection, SimulationResult, TeamRating
from .scheduler import Scheduler
from .scrapers.base import OddsQuote


@dataclasses.dataclass(slots=True)
class CommandContext:
    """Runtime objects shared across command handlers."""

    service: OddsIngestionService
    alert_manager: AlertManager | None


CommandHandler = Callable[[CommandContext, argparse.Namespace], Awaitable[None]]


@dataclasses.dataclass(slots=True)
class CommandSpec:
    """Container describing a CLI sub-command."""

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    handler: CommandHandler


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


def _detect_opportunities(
    quotes: Sequence[OddsQuote],
    simulations: Sequence[SimulationResult],
    *,
    value_threshold: float,
    alert_manager: AlertManager | None,
) -> List[Opportunity]:
    detector = EdgeDetector(
        value_threshold=value_threshold,
        player_model=_build_player_model(),
        alert_manager=alert_manager,
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


def _maybe_alert_ingestion_health(
    service: OddsIngestionService, alert_manager: AlertManager | None
) -> None:
    if alert_manager:
        alert_manager.notify_ingestion_health(service.metrics)


def _create_service(storage_path: str) -> OddsIngestionService:
    return OddsIngestionService([MockSportsbookScraper()], storage_path=storage_path)


def _configure_ingest_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)


def _configure_simulate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument("--value-threshold", type=float, default=0.01)
    parser.add_argument("--history-limit", type=int, default=128)
    parser.add_argument("--movement-threshold", type=int, default=40)


def _configure_scan_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--history-limit", type=int, default=256)
    parser.add_argument("--value-threshold", type=float, default=0.02)
    parser.add_argument("--movement-threshold", type=int, default=30)


def _configure_dashboard_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=15_000)
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument("--value-threshold", type=float, default=0.015)


def _configure_backtest_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--value-threshold", type=float, default=0.02)


async def _cmd_ingest(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    if args.interval <= 0:
        stored = await service.fetch_and_store()
        _maybe_alert_ingestion_health(service, alert_manager)
        print(f"Ingested {len(stored)} quotes into {args.storage}")
        return

    scheduler = Scheduler()

    async def _collect() -> None:
        await service.fetch_and_store()
        _maybe_alert_ingestion_health(service, alert_manager)

    scheduler.add_job(
        _collect,
        interval=args.interval,
        jitter=args.jitter,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        name="odds-ingest",
    )
    install_signal_handlers(scheduler.stop)
    print("Starting ingestion scheduler. Press Ctrl+C to stop.")
    try:
        await scheduler.run()
    finally:
        await scheduler.shutdown()


async def _cmd_simulate(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    if args.refresh:
        ingested = await service.fetch_and_store()
        _maybe_alert_ingestion_health(service, alert_manager)
    else:
        ingested = service.load_latest()
        if not ingested:
            ingested = await service.fetch_and_store()
            _maybe_alert_ingestion_health(service, alert_manager)
    quotes = _quotes_from_ingested(ingested)
    simulations = _run_simulations(quotes, args.iterations)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
    )
    _render_opportunities(opportunities)
    _portfolio_allocation(opportunities, bankroll=args.bankroll)
    movements = _line_movement(
        service,
        limit=args.history_limit,
        alert_manager=alert_manager,
        threshold=args.movement_threshold,
    )
    _print_line_movements(movements)


async def _cmd_scan(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    history = service.load_history(limit=args.history_limit)
    if not history:
        history = await service.fetch_and_store()
        _maybe_alert_ingestion_health(service, alert_manager)
    quotes = _quotes_from_ingested(history)
    simulations = _run_simulations(quotes, args.iterations)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
    )
    _render_opportunities(opportunities)
    movements = _line_movement(
        service,
        limit=args.history_limit,
        alert_manager=alert_manager,
        threshold=args.movement_threshold,
    )
    _print_line_movements(movements)


async def _cmd_dashboard(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    if args.refresh:
        latest = await service.fetch_and_store()
        _maybe_alert_ingestion_health(service, alert_manager)
    else:
        latest = service.load_latest()
        if not latest:
            latest = await service.fetch_and_store()
            _maybe_alert_ingestion_health(service, alert_manager)
    quotes = _quotes_from_ingested(latest)
    simulations = _run_simulations(quotes, args.iterations)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        value_threshold=args.value_threshold,
        alert_manager=alert_manager,
    )
    dashboard = Dashboard()
    rendered = dashboard.render(latest, simulations, opportunities)
    print(rendered)


async def _cmd_backtest(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
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

    commands: list[CommandSpec] = [
        CommandSpec(
            name="ingest",
            help="Collect odds",
            configure=_configure_ingest_parser,
            handler=_cmd_ingest,
        ),
        CommandSpec(
            name="simulate",
            help="Simulate markets and rank opportunities",
            configure=_configure_simulate_parser,
            handler=_cmd_simulate,
        ),
        CommandSpec(
            name="scan",
            help="Scan stored markets for edges and movement",
            configure=_configure_scan_parser,
            handler=_cmd_scan,
        ),
        CommandSpec(
            name="dashboard",
            help="Render the ASCII dashboard",
            configure=_configure_dashboard_parser,
            handler=_cmd_dashboard,
        ),
        CommandSpec(
            name="backtest",
            help="Replay historical odds and summarise expected value",
            configure=_configure_backtest_parser,
            handler=_cmd_backtest,
        ),
    ]

    for command in commands:
        parser_kwargs = dict(parents=[parent], help=command.help)
        subparser = subparsers.add_parser(command.name, **parser_kwargs)
        command.configure(subparser)
        subparser.set_defaults(handler=command.handler)

    return parser


async def _dispatch(args: argparse.Namespace) -> None:
    alert_manager = get_alert_manager(args.alerts_config)
    service = _create_service(args.storage)
    context = CommandContext(service=service, alert_manager=alert_manager)
    handler: CommandHandler = args.handler
    await handler(context, args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_dispatch(args))


__all__ = ["main"]

