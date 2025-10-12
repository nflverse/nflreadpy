"""Command line interface for the Bloomberg-style betting stack."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import json
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)
from typing import runtime_checkable

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
    ConfigurationError,
    create_edge_detector,
    create_ingestion_service,
    load_betting_config,
    validate_betting_config,
)


@dataclasses.dataclass(slots=True)
class CommandContext:
    """Runtime objects shared across command handlers."""

    service: OddsIngestionService
    alert_manager: AlertManager | None
    config: BettingConfig


@runtime_checkable
class ContextCommandHandler(Protocol):
    async def __call__(self, context: CommandContext, args: argparse.Namespace) -> None:
        """Execute a command that relies on a service context."""


@runtime_checkable
class ConfigCommandHandler(Protocol):
    async def __call__(self, config: BettingConfig, args: argparse.Namespace) -> None:
        """Execute a command that only needs configuration data."""


CommandHandler = ContextCommandHandler | ConfigCommandHandler

HandlerT = TypeVar("HandlerT", bound=CommandHandler)


@dataclasses.dataclass(slots=True)
class Subcommand:
    """Container describing a CLI sub-command."""

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    handler: CommandHandler
    requires_service: bool

    def add_to_parser(
        self,
        subparsers,
        parent: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Create the parser for this subcommand."""

        parser = subparsers.add_parser(self.name, parents=[parent], help=self.help)
        self.configure(parser)
        parser.set_defaults(
            handler=self.handler,
            command=self.name,
            requires_service=self.requires_service,
        )
        return parser


class SubcommandApp:
    """Registry that wires handlers into an :class:`argparse` parser."""

    def __init__(self, description: str | None = None) -> None:
        self._commands: list[Subcommand] = []
        self._description = description

    def command(
        self,
        name: str,
        *,
        help: str,
        configure: Callable[[argparse.ArgumentParser], None],
        requires_service: bool = True,
    ) -> Callable[[HandlerT], HandlerT]:
        """Register ``handler`` as a sub-command with configuration callback."""

        def _decorator(handler: HandlerT) -> HandlerT:
            if requires_service and not isinstance(handler, ContextCommandHandler):
                raise TypeError("Service commands must accept a context")
            if not requires_service and not isinstance(handler, ConfigCommandHandler):
                raise TypeError("Config-only commands must accept a configuration object")
            self._commands.append(
                Subcommand(
                    name=name,
                    help=help,
                    configure=configure,
                    handler=handler,
                    requires_service=requires_service,
                )
            )
            return handler

        return _decorator

    @property
    def commands(self) -> Sequence[Subcommand]:
        return tuple(self._commands)

    def build_parser(self) -> argparse.ArgumentParser:
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("--config", dest="config_file")
        parent.add_argument("--environment", dest="config_environment")
        parent.add_argument("--storage")
        parent.add_argument("--alerts-config")

        parser = argparse.ArgumentParser(description=self._description)
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in self._commands:
            command.add_to_parser(subparsers, parent)
        return parser


APP = SubcommandApp(description=__doc__)


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
        for group, mapping_value in values.items():
            try:
                limits[str(group)] = float(mapping_value)
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
    return cast(List[Opportunity], consolidate_best_prices(opportunities))


def _render_opportunities(opportunities: Sequence[Opportunity]) -> None:
    if not opportunities:
        print("No actionable opportunities identified.")
        return
    header = (
        f"{'Event':<16} {'Market':<20} {'Selection':<22} {'US':>6}"
        f" {'Dec':>6} {'Frac':>7} {'Model':>7} {'Imp':>7} {'EV':>8} {'Kelly':>8}"
    )
    print(header)
    print("-" * len(header))
    for opp in opportunities:
        fractional_num, fractional_den = opp.fractional_odds()
        fractional_display = f"{fractional_num}/{fractional_den}"
        decimal = opp.decimal_odds()
        print(
            f"{opp.event_id:<16} {opp.market:<20} {opp.team_or_player:<22}"
            f" {opp.american_odds:>+6d} {decimal:>6.2f} {fractional_display:>7}"
            f" {opp.model_probability:>7.3f} {opp.implied_probability:>7.3f}"
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
    limits = manager.correlation_limits
    if limits:
        print("\nCorrelation limits:")
        for group, fraction in sorted(limits.items()):
            print(f"  {group}: {fraction:.2%} of bankroll")
    if simulation:
        metrics = manager.bankroll_summary() or simulation.summary()
        print("\nBankroll simulation summary:")
        print(f"  Trials: {int(metrics['trials'])}")
        print(f"  Mean terminal: {metrics['mean_terminal']:.2f}")
        print(f"  Median terminal: {metrics['median_terminal']:.2f}")
        print(f"  Worst terminal: {metrics['worst_terminal']:.2f}")
        print(f"  Average drawdown: {metrics['average_drawdown']:.2%}")
        print(f"  Worst drawdown: {metrics['worst_drawdown']:.2%}")
        print(f"  5th percentile drawdown: {metrics['p05_drawdown']:.2%}")
        print(f"  95th percentile drawdown: {metrics['p95_drawdown']:.2%}")
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
    return cast(List[LineMovement], analyzer.summarise())


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


async def _ingest_once(
    service: OddsIngestionService, alert_manager: AlertManager | None
) -> Sequence[IngestedOdds]:
    """Fetch and persist odds data, emitting health alerts when configured."""

    stored = await service.fetch_and_store()
    _maybe_alert_ingestion_health(service, alert_manager)
    return stored


def _maybe_alert_ingestion_health(
    service: OddsIngestionService, alert_manager: AlertManager | None
) -> None:
    if alert_manager:
        alert_manager.notify_ingestion_health(service.metrics)


async def _ensure_latest(
    service: OddsIngestionService,
    alert_manager: AlertManager | None,
    *,
    refresh: bool,
) -> Sequence[IngestedOdds]:
    """Return the freshest stored odds, refreshing via ingestion when required."""

    if refresh:
        return await _ingest_once(service, alert_manager)
    latest = service.load_latest()
    if latest:
        return latest
    return await _ingest_once(service, alert_manager)


async def _ensure_history(
    service: OddsIngestionService,
    alert_manager: AlertManager | None,
    *,
    limit: int,
) -> Sequence[IngestedOdds]:
    """Return historical odds data, triggering ingestion if none are available."""

    history = service.load_history(limit=limit)
    if history:
        return history
    return await _ingest_once(service, alert_manager)


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
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--movement-threshold", type=int)
    parser.add_argument("--movement-depth", type=int)
    parser.add_argument("--stale-after-minutes", type=float)


def _configure_backtest_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--value-threshold", type=float)


def _configure_validate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Fail validation when configuration warnings are encountered.",
    )


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
    if hasattr(args, "movement_depth") and getattr(args, "movement_depth", None) is None:
        args.movement_depth = getattr(analytics, "movement_depth", None)
    if hasattr(args, "stale_after_minutes") and getattr(args, "stale_after_minutes", None) is None:
        args.stale_after_minutes = getattr(analytics, "stale_after_minutes", 10.0)

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


@APP.command(
    "validate-config",
    help="Validate betting configuration",
    configure=_configure_validate_parser,
    requires_service=False,
)
async def _cmd_validate_config(
    config: BettingConfig, args: argparse.Namespace
) -> None:
    try:
        warnings = validate_betting_config(config)
    except ConfigurationError as exc:
        print("Configuration invalid:")
        for line in str(exc).splitlines():
            text = line if line.startswith("-") else f"- {line}"
            print(text)
        raise SystemExit(1) from exc

    print(f"Configuration '{config.environment}' is valid.")
    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"- {message}")
        if getattr(args, "warnings_as_errors", False):
            raise SystemExit(2)


@APP.command(
    "ingest",
    help="Collect odds",
    configure=_configure_ingest_parser,
)
async def _cmd_ingest(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    if args.interval <= 0:
        stored = await _ingest_once(service, alert_manager)
        print(f"Ingested {len(stored)} quotes into {args.storage}")
        return

    async with Scheduler() as scheduler:
        async def _collect() -> None:
            await _ingest_once(service, alert_manager)

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
        await scheduler.run()


@APP.command(
    "simulate",
    help="Simulate markets and rank opportunities",
    configure=_configure_simulate_parser,
)
async def _cmd_simulate(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    ingested = await _ensure_latest(service, alert_manager, refresh=args.refresh)
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


@APP.command(
    "scan",
    help="Scan stored markets for edges and movement",
    configure=_configure_scan_parser,
)
async def _cmd_scan(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    history = await _ensure_history(
        service,
        alert_manager,
        limit=args.history_limit,
    )
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


@APP.command(
    "dashboard",
    help="Render the ASCII dashboard",
    configure=_configure_dashboard_parser,
)
async def _cmd_dashboard(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    latest = await _ensure_latest(service, alert_manager, refresh=args.refresh)
    quotes = _quotes_from_ingested(latest)
    history = await _ensure_history(
        service,
        alert_manager,
        limit=args.history_limit,
    )
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
    if args.movement_depth is not None:
        dashboard.movement_depth = args.movement_depth
    if args.movement_threshold is not None:
        dashboard.movement_threshold = args.movement_threshold
    if args.stale_after_minutes is not None:
        dashboard.stale_after = dt.timedelta(minutes=args.stale_after_minutes)
    risk_summary = RiskSummary(
        bankroll=args.bankroll,
        opportunity_fraction=args.kelly_fraction,
        portfolio_fraction=args.portfolio_fraction,
        positions=tuple(manager.positions),
        exposure_by_event=manager.exposure_report(),
        correlation_exposure=manager.correlation_report(),
        correlation_limits=manager.correlation_limits,
        simulation=simulation,
        bankroll_summary=manager.bankroll_summary(),
    )
    rendered = dashboard.render(
        latest,
        simulations,
        opportunities,
        risk_summary=risk_summary,
        movement_history=history,
        movement_depth=args.movement_depth,
        movement_threshold=args.movement_threshold,
        stale_after=dt.timedelta(minutes=args.stale_after_minutes)
        if args.stale_after_minutes is not None
        else None,
    )
    print(rendered)


@APP.command(
    "backtest",
    help="Replay historical odds and summarise expected value",
    configure=_configure_backtest_parser,
)
async def _cmd_backtest(context: CommandContext, args: argparse.Namespace) -> None:
    service = context.service
    alert_manager = context.alert_manager
    history = service.load_history(limit=args.limit)
    if not history:
        history = await _ensure_history(service, alert_manager, limit=args.limit)
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
    return APP.build_parser()


async def _dispatch(args: argparse.Namespace) -> None:
    config = load_betting_config(
        base_path=args.config_file,
        environment=args.config_environment,
    )
    requires_service = getattr(args, "requires_service", True)
    handler: CommandHandler = args.handler
    if isinstance(handler, ConfigCommandHandler):
        await handler(config, args)
        return

    try:
        warnings = validate_betting_config(config)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    if warnings:
        for message in warnings:
            print(f"[config-warning] {message}")

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
    if not isinstance(handler, ContextCommandHandler):  # pragma: no cover - safety net
        raise TypeError("Command handler does not support service context")
    await handler(context, args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_dispatch(args))


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

