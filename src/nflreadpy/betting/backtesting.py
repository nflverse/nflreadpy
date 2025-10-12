"""Historical odds backtesting utilities for sportsbook snapshots.

This module loads historical open, mid, and closing snapshots, applies
book-specific settlement rules, and derives profitability and calibration
metrics that can be surfaced in dashboards.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import polars as pl

RowMapping = Mapping[str, Any]


__all__ = [
    "BacktestArtifacts",
    "BacktestMetrics",
    "QuantumComparisonArtifacts",
    "QuantumScenarioComparison",
    "ScenarioPerformance",
    "Settlement",
    "SportsbookRules",
    "closing_line_table",
    "export_closing_line_report",
    "export_reliability_diagram",
    "get_sportsbook_rules",
    "load_historical_snapshots",
    "persist_quantum_comparison",
    "persist_backtest_reports",
    "reliability_table",
    "compare_quantum_backtest",
    "run_backtest",
    "settlements_to_frame",
    "simulate_settlements",
]

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .analytics import Opportunity as OpportunityType, PortfolioPosition as PortfolioPositionType


@dataclasses.dataclass(frozen=True, slots=True)
class SportsbookRules:
    """Encapsulate sportsbook-specific settlement behaviour."""

    name: str
    three_way_moneyline: bool = False
    push_on_spread: bool = True
    push_on_total: bool = True


# A lightweight registry of sportsbook specific overrides.  Additional books can
# be registered over time as we encounter differences in settlement behaviour.
DEFAULT_SPORTSBOOK_RULES: Mapping[str, SportsbookRules] = {
    "default": SportsbookRules(name="default"),
    "testbook": SportsbookRules(name="testbook"),
}


def get_sportsbook_rules(
    sportsbook: str,
    overrides: Mapping[str, SportsbookRules] | None = None,
) -> SportsbookRules:
    """Return the settlement rules for ``sportsbook``.

    Parameters
    ----------
    sportsbook:
        Sportsbook name, case insensitive.
    overrides:
        Optional mapping of sportsbook names to :class:`SportsbookRules` which
        take precedence over the built-in defaults.
    """

    key = sportsbook.lower()
    if overrides is not None and key in overrides:
        return overrides[key]
    if key in DEFAULT_SPORTSBOOK_RULES:
        return DEFAULT_SPORTSBOOK_RULES[key]
    return DEFAULT_SPORTSBOOK_RULES["default"]


@dataclasses.dataclass(slots=True)
class Settlement:
    """Result of settling a single historical odds snapshot."""

    event_id: str
    sportsbook: str
    team_or_player: str
    market: str
    scope: str
    snapshot_type: str | None
    side: str | None
    line: float | None
    american_odds: int
    stake: float
    model_probability: float
    outcome: str
    actual_outcome: float
    pnl: float
    brier: float
    log_loss: float
    crps: float
    closing_american_odds: int | None
    closing_line: float | None


@dataclasses.dataclass(slots=True)
class BacktestMetrics:
    """Aggregated metrics computed from a sequence of settlements."""

    total_pnl: float
    average_brier: float
    average_log_loss: float
    average_crps: float
    settlements: Sequence[Settlement]


@dataclasses.dataclass(frozen=True, slots=True)
class BacktestArtifacts:
    """Persisted artefacts produced by a backtest run."""

    metrics: BacktestMetrics
    reliability_path: Path
    closing_line_path: Path


@dataclasses.dataclass(slots=True)
class ScenarioPerformance:
    """Summary metrics for a single staking strategy."""

    name: str
    expected_value: float
    realized_pnl: float
    hit_rate: float
    wins: int
    attempts: int
    average_drawdown: float
    max_drawdown: float
    per_event_profits: list[float] = dataclasses.field(repr=False)
    per_event_expected: list[float] = dataclasses.field(repr=False)


@dataclasses.dataclass(slots=True)
class QuantumScenarioComparison:
    """Comparison between baseline and quantum-optimised staking."""

    baseline: ScenarioPerformance
    quantum: ScenarioPerformance
    delta_expected_value: float
    delta_hit_rate: float
    delta_max_drawdown: float
    profit_differences: list[float] = dataclasses.field(repr=False)
    t_statistic: float | None = None
    t_p_value: float | None = None
    bootstrap_p_value: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class QuantumComparisonArtifacts:
    """File artefacts produced when persisting quantum comparison results."""

    summary_path: Path
    distribution_path: Path
    significance_path: Path


def load_historical_snapshots(path: str | Path) -> pl.DataFrame:
    """Load historical odds snapshots stored as CSV or Parquet files.

    Parameters
    ----------
    path:
        Path to a single file or a directory containing snapshot files. Files
        must be in CSV or Parquet format and share a compatible schema.
    """

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Snapshot path does not exist: {target}")

    files: list[Path]
    if target.is_file():
        files = [target]
    else:
        files = sorted(
            [
                candidate
                for candidate in target.rglob("*")
                if candidate.suffix.lower() in {".csv", ".parquet"}
            ]
        )
    if not files:
        raise ValueError(f"No snapshot files found under {target}")

    frames: list[pl.DataFrame] = []
    for file in files:
        if file.suffix.lower() == ".csv":
            frames.append(pl.read_csv(file))
        elif file.suffix.lower() == ".parquet":
            frames.append(pl.read_parquet(file))
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported snapshot format: {file.suffix}")

    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical_relaxed")


def run_backtest(
    snapshots: pl.DataFrame,
    *,
    sportsbook_rules: Mapping[str, SportsbookRules] | None = None,
) -> BacktestMetrics:
    """Simulate settlements and compute summary metrics."""

    settlements = list(
        simulate_settlements(snapshots, sportsbook_rules=sportsbook_rules)
    )
    if not settlements:
        raise ValueError("No settlements produced from snapshots")

    total_pnl = sum(item.pnl for item in settlements)
    average_brier = sum(item.brier for item in settlements) / len(settlements)
    average_log_loss = sum(item.log_loss for item in settlements) / len(settlements)
    average_crps = sum(item.crps for item in settlements) / len(settlements)
    return BacktestMetrics(
        total_pnl=total_pnl,
        average_brier=average_brier,
        average_log_loss=average_log_loss,
        average_crps=average_crps,
        settlements=settlements,
    )


def simulate_settlements(
    snapshots: pl.DataFrame,
    *,
    sportsbook_rules: Mapping[str, SportsbookRules] | None = None,
) -> Iterable[Settlement]:
    """Yield :class:`Settlement` objects for each odds snapshot."""

    required_columns = {
        "event_id",
        "sportsbook",
        "market",
        "scope",
        "entity_type",
        "team_or_player",
        "side",
        "line",
        "american_odds",
        "model_probability",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing = required_columns.difference(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshots missing required columns: {sorted(missing)}")

    for row in snapshots.iter_rows(named=True):
        settlement = _settle_row(row, sportsbook_rules=sportsbook_rules)
        yield settlement


def export_reliability_diagram(
    settlements: Sequence[Settlement],
    output_path: str | Path,
    bins: int = 10,
) -> Path:
    """Generate a reliability diagram table and persist it as CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = reliability_table(settlements, bins=bins)
    table.write_csv(output)
    return output


def export_closing_line_report(
    settlements: Sequence[Settlement],
    output_path: str | Path,
) -> Path:
    """Persist a closing line comparison report as CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = closing_line_table(settlements)
    table.write_csv(output)
    return output


def reliability_table(
    settlements: Sequence[Settlement],
    bins: int = 10,
) -> pl.DataFrame:
    """Return the reliability table used for calibration diagnostics."""

    return _reliability_table(settlements, bins)


def closing_line_table(settlements: Sequence[Settlement]) -> pl.DataFrame:
    """Return the closing line comparison table."""

    return _closing_line_table(settlements)


def persist_backtest_reports(
    metrics: BacktestMetrics,
    output_dir: str | Path,
    *,
    bins: int = 10,
    reliability_filename: str = "reliability_diagram.csv",
    closing_line_filename: str = "closing_line_report.csv",
) -> BacktestArtifacts:
    """Persist derived artefacts from a backtest run.

    Parameters
    ----------
    metrics:
        The metrics returned by :func:`run_backtest`.
    output_dir:
        Directory where the generated files will be stored.
    bins:
        Number of bins to use when building the reliability diagram table.
    reliability_filename:
        Name of the CSV file containing the reliability diagram data.
    closing_line_filename:
        Name of the CSV file containing the closing line efficiency report.
    """

    settlements = metrics.settlements
    if not settlements:
        raise ValueError("Cannot persist reports without settlements")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    reliability_path = destination / reliability_filename
    closing_line_path = destination / closing_line_filename

    export_reliability_diagram(settlements, reliability_path, bins=bins)
    export_closing_line_report(settlements, closing_line_path)

    return BacktestArtifacts(
        metrics=metrics,
        reliability_path=reliability_path,
        closing_line_path=closing_line_path,
    )


def compare_quantum_backtest(
    snapshots: pl.DataFrame,
    *,
    sportsbook_rules: Mapping[str, SportsbookRules] | None = None,
    bankroll: float = 1_000.0,
    max_risk_per_bet: float = 0.02,
    max_event_exposure: float = 0.1,
    portfolio_fraction: float = 1.0,
    optimizer: "QuantumPortfolioOptimizer | None" = None,
    optimizer_shots: int = 512,
    optimizer_temperature: float = 0.6,
    optimizer_seed: int | None = None,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int | None = None,
) -> QuantumScenarioComparison:
    """Compare baseline and quantum-optimised staking on historical snapshots."""

    settlements = list(
        simulate_settlements(snapshots, sportsbook_rules=sportsbook_rules)
    )
    if not settlements:
        raise ValueError("No settlements available for comparison")

    opportunities = _snapshots_to_opportunities(snapshots)
    if not opportunities:
        raise ValueError("No opportunities available for quantum optimisation")

    baseline_order = sorted(
        opportunities, key=lambda opportunity: opportunity.expected_value, reverse=True
    )
    baseline_positions = _allocate_portfolio_positions(
        baseline_order,
        bankroll=bankroll,
        max_risk_per_bet=max_risk_per_bet,
        max_event_exposure=max_event_exposure,
        portfolio_fraction=portfolio_fraction,
    )
    baseline_stakes = _positions_to_stake_map(baseline_positions)

    if optimizer is None:
        from .quantum import QuantumPortfolioOptimizer  # pragma: no cover - optional import

        optimizer = QuantumPortfolioOptimizer(
            shots=optimizer_shots,
            temperature=optimizer_temperature,
            seed=optimizer_seed,
        )

    ranked = optimizer.optimise(opportunities)
    quantum_order = [item[0] for item in ranked]
    seen_ids = {id(opportunity) for opportunity in quantum_order}
    quantum_order.extend(
        opportunity for opportunity in opportunities if id(opportunity) not in seen_ids
    )

    quantum_positions = _allocate_portfolio_positions(
        quantum_order,
        bankroll=bankroll,
        max_risk_per_bet=max_risk_per_bet,
        max_event_exposure=max_event_exposure,
        portfolio_fraction=portfolio_fraction,
    )
    quantum_stakes = _positions_to_stake_map(quantum_positions)

    rows = list(snapshots.iter_rows(named=True))
    baseline_perf = _scenario_performance(
        "baseline",
        rows,
        settlements,
        baseline_stakes,
    )
    quantum_perf = _scenario_performance(
        "quantum",
        rows,
        settlements,
        quantum_stakes,
    )

    profit_diffs = [
        quantum - baseline
        for baseline, quantum in zip(
            baseline_perf.per_event_profits, quantum_perf.per_event_profits
        )
    ]
    t_stat, t_p = _paired_t_test(
        baseline_perf.per_event_profits, quantum_perf.per_event_profits
    )
    bootstrap_p = _bootstrap_p_value(
        profit_diffs, iterations=bootstrap_iterations, seed=bootstrap_seed
    )

    return QuantumScenarioComparison(
        baseline=baseline_perf,
        quantum=quantum_perf,
        delta_expected_value=quantum_perf.expected_value - baseline_perf.expected_value,
        delta_hit_rate=quantum_perf.hit_rate - baseline_perf.hit_rate,
        delta_max_drawdown=quantum_perf.max_drawdown - baseline_perf.max_drawdown,
        profit_differences=profit_diffs,
        t_statistic=t_stat,
        t_p_value=t_p,
        bootstrap_p_value=bootstrap_p,
    )


def persist_quantum_comparison(
    comparison: QuantumScenarioComparison,
    output_dir: str | Path,
    *,
    summary_filename: str = "summary.csv",
    distribution_filename: str = "distribution.csv",
    significance_filename: str = "significance.json",
) -> QuantumComparisonArtifacts:
    """Persist summary artefacts describing quantum optimisation impact."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_records = [
        {
            "scenario": comparison.baseline.name,
            "expected_value": comparison.baseline.expected_value,
            "realized_pnl": comparison.baseline.realized_pnl,
            "hit_rate": comparison.baseline.hit_rate,
            "wins": comparison.baseline.wins,
            "attempts": comparison.baseline.attempts,
            "average_drawdown": comparison.baseline.average_drawdown,
            "max_drawdown": comparison.baseline.max_drawdown,
        },
        {
            "scenario": comparison.quantum.name,
            "expected_value": comparison.quantum.expected_value,
            "realized_pnl": comparison.quantum.realized_pnl,
            "hit_rate": comparison.quantum.hit_rate,
            "wins": comparison.quantum.wins,
            "attempts": comparison.quantum.attempts,
            "average_drawdown": comparison.quantum.average_drawdown,
            "max_drawdown": comparison.quantum.max_drawdown,
        },
        {
            "scenario": "delta",
            "expected_value": comparison.delta_expected_value,
            "realized_pnl": comparison.quantum.realized_pnl
            - comparison.baseline.realized_pnl,
            "hit_rate": comparison.delta_hit_rate,
            "wins": comparison.quantum.wins - comparison.baseline.wins,
            "attempts": comparison.quantum.attempts - comparison.baseline.attempts,
            "average_drawdown": comparison.quantum.average_drawdown
            - comparison.baseline.average_drawdown,
            "max_drawdown": comparison.delta_max_drawdown,
        },
    ]
    summary_path = destination / summary_filename
    pl.DataFrame(summary_records).write_csv(summary_path)

    distribution_records = []
    for index, (baseline_profit, quantum_profit, diff) in enumerate(
        zip(
            comparison.baseline.per_event_profits,
            comparison.quantum.per_event_profits,
            comparison.profit_differences,
        )
    ):
        distribution_records.append(
            {
                "event_index": index,
                "scenario": comparison.baseline.name,
                "profit": baseline_profit,
            }
        )
        distribution_records.append(
            {
                "event_index": index,
                "scenario": comparison.quantum.name,
                "profit": quantum_profit,
            }
        )
        distribution_records.append(
            {
                "event_index": index,
                "scenario": "difference",
                "profit": diff,
            }
        )
    distribution_path = destination / distribution_filename
    pl.DataFrame(distribution_records).write_csv(distribution_path)

    significance_payload = {
        "t_statistic": comparison.t_statistic,
        "t_p_value": comparison.t_p_value,
        "bootstrap_p_value": comparison.bootstrap_p_value,
        "delta_expected_value": comparison.delta_expected_value,
        "delta_hit_rate": comparison.delta_hit_rate,
        "delta_max_drawdown": comparison.delta_max_drawdown,
    }
    significance_path = destination / significance_filename
    significance_path.write_text(json.dumps(significance_payload, indent=2, sort_keys=True))

    return QuantumComparisonArtifacts(
        summary_path=summary_path,
        distribution_path=distribution_path,
        significance_path=significance_path,
    )


def _snapshots_to_opportunities(
    snapshots: pl.DataFrame,
) -> list["OpportunityType"]:
    from .analytics import KellyCriterion, Opportunity

    opportunities: list["OpportunityType"] = []
    for row in snapshots.iter_rows(named=True):
        price = int(row["american_odds"])
        win_prob = float(row["model_probability"])
        win_prob = max(0.0, min(1.0, win_prob))
        loss_prob = max(0.0, 1.0 - win_prob)
        expected_value = win_prob * _profit_multiplier(price) - loss_prob
        kelly_fraction = KellyCriterion.fraction(
            win_probability=win_prob,
            loss_probability=loss_prob,
            price=price,
        )
        side = row.get("side")
        line = row.get("line")
        snapshot_type = row.get("snapshot_type")
        extra: dict[str, object] = {}
        if snapshot_type is not None:
            extra["snapshot_type"] = str(snapshot_type)
        opportunities.append(
            Opportunity(
                event_id=str(row["event_id"]),
                sportsbook=str(row["sportsbook"]),
                book_market_group=str(row["book_market_group"]),
                market=str(row["market"]),
                scope=str(row["scope"]),
                entity_type=str(row["entity_type"]),
                team_or_player=str(row["team_or_player"]),
                side=None if side is None else str(side),
                line=None if line is None else float(line),
                american_odds=price,
                model_probability=win_prob,
                push_probability=0.0,
                implied_probability=_implied_probability(price),
                expected_value=expected_value,
                kelly_fraction=kelly_fraction,
                extra=extra,
            )
        )
    return opportunities


def _allocate_portfolio_positions(
    opportunities: Sequence["OpportunityType"],
    *,
    bankroll: float,
    max_risk_per_bet: float,
    max_event_exposure: float,
    portfolio_fraction: float,
) -> Sequence["PortfolioPositionType"]:
    from .analytics import PortfolioManager

    manager = PortfolioManager(
        bankroll=bankroll,
        max_risk_per_bet=max_risk_per_bet,
        max_event_exposure=max_event_exposure,
        fractional_kelly=portfolio_fraction,
    )
    for opportunity in opportunities:
        manager.allocate(opportunity)
    return tuple(manager.positions)


def _positions_to_stake_map(
    positions: Sequence["PortfolioPositionType"],
) -> dict[tuple[str, str, str, str, str | None, float | None], float]:
    stakes: dict[tuple[str, str, str, str, str | None, float | None], float] = {}
    for position in positions:
        opportunity = position.opportunity
        key = (
            opportunity.event_id,
            opportunity.sportsbook,
            opportunity.market,
            opportunity.team_or_player,
            opportunity.side,
            opportunity.line,
        )
        stakes[key] = stakes.get(key, 0.0) + float(position.stake)
    return stakes


def _settlement_key(
    settlement: Settlement,
) -> tuple[str, str, str, str, str | None, float | None]:
    return (
        settlement.event_id,
        settlement.sportsbook,
        settlement.market,
        settlement.team_or_player,
        settlement.side,
        settlement.line,
    )


def _scenario_performance(
    name: str,
    rows: Sequence[RowMapping],
    settlements: Sequence[Settlement],
    stake_map: Mapping[tuple[str, str, str, str, str | None, float | None], float],
) -> ScenarioPerformance:
    profits: list[float] = []
    expected_values: list[float] = []
    wins = 0
    attempts = 0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    drawdown_sum = 0.0
    drawdown_count = 0

    for row, settlement in zip(rows, settlements):
        key = _settlement_key(settlement)
        stake = float(stake_map.get(key, 0.0))
        multiplier = _profit_multiplier(settlement.american_odds)
        win_prob = float(row.get("model_probability", settlement.model_probability))
        win_prob = max(0.0, min(1.0, win_prob))
        loss_prob = max(0.0, 1.0 - win_prob)
        expected_profit = stake * (win_prob * multiplier - loss_prob)
        expected_values.append(expected_profit)

        if stake <= 0.0:
            profits.append(0.0)
            continue

        profit = _payout(settlement.american_odds, stake, settlement.outcome)
        profits.append(profit)
        cumulative += profit
        if cumulative > peak:
            peak = cumulative
        if peak > 0.0:
            drawdown = (peak - cumulative) / peak
            drawdown_sum += drawdown
            drawdown_count += 1
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        if settlement.outcome != "push":
            attempts += 1
            if settlement.outcome == "win":
                wins += 1

    realized_pnl = sum(profits)
    expected_value = sum(expected_values)
    average_drawdown = drawdown_sum / drawdown_count if drawdown_count else 0.0
    hit_rate = wins / attempts if attempts else 0.0
    return ScenarioPerformance(
        name=name,
        expected_value=expected_value,
        realized_pnl=realized_pnl,
        hit_rate=hit_rate,
        wins=wins,
        attempts=attempts,
        average_drawdown=average_drawdown,
        max_drawdown=max_drawdown,
        per_event_profits=profits,
        per_event_expected=expected_values,
    )


def _paired_t_test(
    baseline: Sequence[float], quantum: Sequence[float]
) -> tuple[float | None, float | None]:
    if len(baseline) != len(quantum):
        raise ValueError("Paired samples must have matching lengths")
    if len(baseline) < 2:
        return None, None
    differences = [q - b for b, q in zip(baseline, quantum)]
    mean_diff = statistics.mean(differences)
    try:
        stdev = statistics.stdev(differences)
    except statistics.StatisticsError:
        stdev = 0.0
    if stdev == 0.0:
        if math.isclose(mean_diff, 0.0, abs_tol=1e-12):
            return 0.0, 1.0
        return (math.copysign(math.inf, mean_diff), 0.0)
    n = len(differences)
    t_stat = mean_diff / (stdev / math.sqrt(n))
    cdf = _student_t_cdf(abs(t_stat), n - 1)
    p_value = max(0.0, min(1.0, 2.0 * (1.0 - cdf)))
    return t_stat, p_value


def _student_t_cdf(t_value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive")
    x = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    incomplete_beta = math.betainc(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - 0.5 * incomplete_beta


def _bootstrap_p_value(
    differences: Sequence[float],
    *,
    iterations: int = 2_000,
    seed: int | None = None,
) -> float | None:
    if iterations <= 0:
        return None
    samples = list(differences)
    if not samples:
        return None
    observed = statistics.mean(samples)
    rng = random.Random(seed)
    n = len(samples)
    extreme = 0
    for _ in range(iterations):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        if abs(statistics.mean(resample)) >= abs(observed):
            extreme += 1
    return min(1.0, (extreme + 1) / (iterations + 1))


def _profit_multiplier(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if american_odds > 0:
        return american_odds / 100.0
    return 100.0 / abs(american_odds)


def _settle_row(
    row: RowMapping,
    *,
    sportsbook_rules: Mapping[str, SportsbookRules] | None = None,
) -> Settlement:
    market = str(row["market"]).lower()
    scope = str(row["scope"])
    snapshot_type = row.get("snapshot_type")
    if snapshot_type is not None:
        snapshot_type = str(snapshot_type)
    side = row.get("side")
    if side is not None:
        side = str(side)

    team_or_player = str(row["team_or_player"])
    sportsbook = str(row["sportsbook"])
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    home_score = float(row["home_score"])
    away_score = float(row["away_score"])
    line = row.get("line")
    line_value = None if line is None else float(line)
    stake_value = float(row.get("stake", 1.0))
    price = int(row["american_odds"])
    closing_price = row.get("closing_american_odds")
    closing_line = row.get("closing_line")
    closing_price_value = None if closing_price is None else int(closing_price)
    closing_line_value = None if closing_line is None else float(closing_line)
    probability = float(row["model_probability"])
    rules = get_sportsbook_rules(sportsbook, overrides=sportsbook_rules)

    actual_outcome, state = _resolve_outcome(
        market,
        team_or_player,
        side,
        line_value,
        home_team,
        away_team,
        home_score,
        away_score,
        rules,
    )
    payout = _payout(price, stake_value, state)
    brier = (probability - actual_outcome) ** 2
    log_loss = _log_loss(probability, actual_outcome)
    # For binary outcomes the Continuous Ranked Probability Score equals the Brier score
    crps = brier

    return Settlement(
        event_id=str(row["event_id"]),
        sportsbook=sportsbook,
        team_or_player=team_or_player,
        market=market,
        scope=scope,
        snapshot_type=snapshot_type,
        side=side,
        line=line_value,
        american_odds=price,
        stake=stake_value,
        model_probability=probability,
        outcome=state,
        actual_outcome=actual_outcome,
        pnl=payout,
        brier=brier,
        log_loss=log_loss,
        crps=crps,
        closing_american_odds=closing_price_value,
        closing_line=closing_line_value,
    )


def _resolve_outcome(
    market: str,
    team_or_player: str,
    side: str | None,
    line: float | None,
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
    rules: SportsbookRules,
) -> tuple[float, str]:
    market_key = market.lower()
    if market_key in {"moneyline", "winner"}:
        if math.isclose(home_score, away_score, abs_tol=1e-9):
            if rules.three_way_moneyline:
                return 0.0, "loss"
            return 0.5, "push"
        winner = home_team if home_score > away_score else away_team
        result = 1.0 if team_or_player == winner else 0.0
        return result, "win" if result == 1.0 else "loss"
    if market_key.startswith("spread"):
        if line is None:
            raise ValueError("Spread markets require a line value")
        margin = _team_margin(team_or_player, home_team, away_team, home_score, away_score)
        adjusted = margin + line
        if math.isclose(adjusted, 0.0, abs_tol=1e-9):
            if rules.push_on_spread:
                return 0.5, "push"
            return 0.0, "loss"
        return (1.0, "win") if adjusted > 0 else (0.0, "loss")
    if market_key in {"total", "game_total"}:
        if line is None or side is None:
            raise ValueError("Total markets require both line and side")
        total_points = home_score + away_score
        if side.lower() == "over":
            diff = total_points - line
        else:
            diff = line - total_points
        if math.isclose(diff, 0.0, abs_tol=1e-9):
            if rules.push_on_total:
                return 0.5, "push"
            return 0.0, "loss"
        return (1.0, "win") if diff > 0 else (0.0, "loss")
    raise NotImplementedError(f"Unsupported market type: {market}")


def _team_margin(
    team_or_player: str,
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
) -> float:
    if team_or_player == home_team:
        return home_score - away_score
    if team_or_player == away_team:
        return away_score - home_score
    raise ValueError("Team did not participate in event")


def _payout(american_odds: int, stake: float, state: str) -> float:
    if state == "push":
        return 0.0
    if state not in {"win", "loss"}:
        raise ValueError(f"Unknown settlement state: {state}")
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if state == "loss":
        return -stake
    if american_odds > 0:
        return stake * (american_odds / 100.0)
    return stake * (100.0 / abs(american_odds))


def _log_loss(probability: float, outcome: float) -> float:
    clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(
        outcome * math.log(clipped)
        + (1.0 - outcome) * math.log(1.0 - clipped)
    )


def _reliability_table(settlements: Sequence[Settlement], bins: int) -> pl.DataFrame:
    if bins <= 0:
        raise ValueError("Number of bins must be positive")
    edges = [i / bins for i in range(bins + 1)]
    buckets: dict[int, list[tuple[float, float]]] = {i: [] for i in range(bins)}
    for settlement in settlements:
        probability = settlement.model_probability
        outcome = settlement.actual_outcome
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Model probabilities must be between 0 and 1")
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, outcome))

    records = []
    for index, values in buckets.items():
        if not values:
            continue
        probs, outcomes = zip(*values)
        records.append(
            {
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "count": len(values),
                "predicted_mean": sum(probs) / len(values),
                "observed_rate": sum(outcomes) / len(values),
            }
        )
    return pl.DataFrame(records)


def _closing_line_table(settlements: Sequence[Settlement]) -> pl.DataFrame:
    records = []
    for settlement in settlements:
        closing_odds = settlement.closing_american_odds
        if closing_odds is None and settlement.closing_line is None:
            continue
        initial_implied = _implied_probability(settlement.american_odds)
        closing_implied = (
            _implied_probability(closing_odds) if closing_odds is not None else None
        )
        model_edge = settlement.model_probability - initial_implied
        closing_model_edge = (
            settlement.model_probability - closing_implied
            if closing_implied is not None
            else None
        )
        records.append(
            {
                "event_id": settlement.event_id,
                "team_or_player": settlement.team_or_player,
                "market": settlement.market,
                "scope": settlement.scope,
                "american_odds": settlement.american_odds,
                "closing_american_odds": closing_odds,
                "odds_delta": None
                if closing_odds is None
                else closing_odds - settlement.american_odds,
                "implied_probability": initial_implied,
                "closing_implied_probability": closing_implied,
                "implied_delta": None
                if closing_implied is None
                else closing_implied - initial_implied,
                "model_probability": settlement.model_probability,
                "model_edge": model_edge,
                "closing_model_edge": closing_model_edge,
                "edge_delta": None
                if closing_model_edge is None
                else closing_model_edge - model_edge,
                "line": settlement.line,
                "closing_line": settlement.closing_line,
                "line_delta": None
                if settlement.closing_line is None or settlement.line is None
                else settlement.closing_line - settlement.line,
            }
        )
    return pl.DataFrame(records)


def _implied_probability(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def settlements_to_frame(settlements: Sequence[Settlement]) -> pl.DataFrame:
    """Convert a collection of settlements into a Polars DataFrame."""

    return pl.DataFrame([dataclasses.asdict(item) for item in settlements])

