"""Command line interface for the Bloomberg-style betting stack."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import (
    Dashboard,
    GameSimulationConfig,
    LineMovementAnalyzer,
    MonteCarloEngine,
    OddsIngestionService,
    Opportunity,
    PlayerPropForecaster,
    PortfolioManager,
    BankrollSimulationResult,
    QuantumPortfolioOptimizer,
    consolidate_best_prices,
)
from .alerts import AlertManager, get_alert_manager, install_signal_handlers
from .analytics import LineMovement
from .dashboard import RiskSummary
from .ingestion import IngestedOdds
from .models import PlayerProjection, SimulationResult, TeamRating
from .scheduler import Scheduler
from .scrapers.base import OddsQuote
from .configuration import (
    BettingConfig,
    create_edge_detector,
    create_ingestion_service,
    load_betting_config,
)


@dataclasses.dataclass(slots=True)
class CommandContext:
    """Runtime objects shared across command handlers."""

    service: OddsIngestionService
    alert_manager: AlertManager | None
    config: BettingConfig


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


def _parse_correlation_limits(
    values: Sequence[str] | Mapping[str, float] | None,
) -> Dict[str, float]:
    limits: Dict[str, float] = {}
    if values is None:
        return limits
    if isinstance(values, Mapping):
        for group, raw_value in values.items():
            try:
                limits[str(group)] = float(raw_value)
            except (TypeError, ValueError):
                continue
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
    config: BettingConfig,
    value_threshold: float,
    alert_manager: AlertManager | None,
    kelly_fraction: float | None = None,
) -> List[Opportunity]:
    detector = create_edge_detector(config, alert_manager=alert_manager)
    detector.value_threshold = value_threshold
    detector.player_model = _build_player_model()
    opportunities = detector.detect(quotes, simulations)
    fractional = config.analytics.kelly_fraction if kelly_fraction is None else kelly_fraction
    if fractional != 1.0:
        opportunities = [
            dataclasses.replace(opp, kelly_fraction=opp.kelly_fraction * fractional)
            for opp in opportunities
        ]
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


def _maybe_alert_ingestion_health(
    service: OddsIngestionService, alert_manager: AlertManager | None
) -> None:
    if alert_manager:
        alert_manager.notify_ingestion_health(service.metrics)


def _create_service(
    config: BettingConfig,
    *,
    storage_path: str,
    alert_manager: AlertManager | None,
) -> OddsIngestionService:
    return create_ingestion_service(
        config,
        storage_path=storage_path,
        alert_sink=alert_manager,
    )


def _configure_ingest_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval", type=float)
    parser.add_argument("--jitter", type=float)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--retry-backoff", type=float)


def _configure_simulate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--bankroll", type=float)
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument("--value-threshold", type=float)
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--movement-threshold", type=int)
    parser.add_argument("--kelly-fraction", type=float)
    parser.add_argument("--portfolio-fraction", type=float)
    parser.add_argument("--correlation-limit", action="append")
    parser.add_argument("--risk-trials", type=int)
    parser.add_argument("--risk-seed", type=int)


def _configure_scan_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--value-threshold", type=float)
    parser.add_argument("--movement-threshold", type=int)
    parser.add_argument("--kelly-fraction", type=float)
    parser.add_argument("--correlation-limit", action="append")


def _configure_dashboard_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument("--value-threshold", type=float)
    parser.add_argument("--bankroll", type=float)
    parser.add_argument("--portfolio-fraction", type=float)
    parser.add_argument("--kelly-fraction", type=float)
    parser.add_argument("--correlation-limit", action="append")
    parser.add_argument("--risk-trials", type=int)
    parser.add_argument("--risk-seed", type=int)


def _configure_backtest_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--value-threshold", type=float)


def _apply_config_defaults(args: argparse.Namespace, config: BettingConfig) -> None:
    scheduler = config.ingestion.scheduler
    if hasattr(args, "interval") and args.interval is None:
        args.interval = scheduler.interval_seconds
    if hasattr(args, "jitter") and args.jitter is None:
        args.jitter = scheduler.jitter_seconds
    if hasattr(args, "retries") and args.retries is None:
        args.retries = scheduler.retries
    if hasattr(args, "retry_backoff") and args.retry_backoff is None:
        args.retry_backoff = scheduler.retry_backoff

    analytics = config.analytics
    iterations = getattr(analytics.iterations, getattr(args, "command", ""), None)
    if hasattr(args, "iterations") and args.iterations is None and iterations is not None:
        args.iterations = iterations
    if hasattr(args, "value_threshold") and args.value_threshold is None:
        args.value_threshold = analytics.value_threshold
    if hasattr(args, "history_limit") and args.history_limit is None:
        args.history_limit = analytics.history_limit
    if hasattr(args, "movement_threshold") and args.movement_threshold is None:
        args.movement_threshold = analytics.movement_threshold

    bankroll = getattr(args, "bankroll", None)
    if bankroll is None and hasattr(args, "bankroll"):
        args.bankroll = analytics.bankroll
    elif not hasattr(args, "bankroll"):
        setattr(args, "bankroll", analytics.bankroll)

    portfolio_fraction = getattr(args, "portfolio_fraction", None)
    if portfolio_fraction is None and hasattr(args, "portfolio_fraction"):
        args.portfolio_fraction = analytics.portfolio_fraction
    elif not hasattr(args, "portfolio_fraction"):
        setattr(args, "portfolio_fraction", analytics.portfolio_fraction)

    kelly_fraction = getattr(args, "kelly_fraction", None)
    if kelly_fraction is None and hasattr(args, "kelly_fraction"):
        args.kelly_fraction = analytics.kelly_fraction
    elif not hasattr(args, "kelly_fraction"):
        setattr(args, "kelly_fraction", analytics.kelly_fraction)

    if not hasattr(args, "correlation_limit") or args.correlation_limit is None:
        if analytics.correlation_limits:
            setattr(args, "correlation_limit", dict(analytics.correlation_limits))

    if hasattr(args, "risk_trials") and args.risk_trials is None:
        args.risk_trials = analytics.risk_trials
    if hasattr(args, "risk_seed") and args.risk_seed is None:
        args.risk_seed = analytics.risk_seed

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
    correlation_limits = _parse_correlation_limits(args.correlation_limit)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        config=context.config,
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
        config=context.config,
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
    correlation_limits = _parse_correlation_limits(args.correlation_limit)
    opportunities = _detect_opportunities(
        quotes,
        simulations,
        config=context.config,
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
            config=context.config,
            value_threshold=args.value_threshold,
            alert_manager=alert_manager,
            kelly_fraction=getattr(args, "kelly_fraction", None),
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
    parent.add_argument("--config", dest="config_file")
    parent.add_argument("--environment", dest="config_environment")
    parent.add_argument("--storage")
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
    config = load_betting_config(
        base_path=args.config_file,
        environment=args.config_environment,
    )
    alert_manager = get_alert_manager(args.alerts_config)
    storage_path = str(args.storage or config.ingestion.storage_path)
    args.storage = storage_path
    _apply_config_defaults(args, config)
    service = _create_service(
        config,
        storage_path=storage_path,
        alert_manager=alert_manager,
    )
    context = CommandContext(service=service, alert_manager=alert_manager, config=config)
    handler: CommandHandler = args.handler
    await handler(context, args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_dispatch(args))


__all__ = ["main"]

