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
    "comparison_performance_table",
    "Settlement",
    "SportsbookRules",
    "closing_line_table",
    "load_comparison_history",
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


def load_comparison_history(path: str | Path) -> pl.DataFrame:
    """Load persisted optimizer comparison allocations."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
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
        raise ValueError(f"No comparison artefacts found under {target}")
    frames: list[pl.DataFrame] = []
    for file in files:
        if file.suffix.lower() == ".csv":
            frames.append(pl.read_csv(file))
        elif file.suffix.lower() == ".parquet":
            frames.append(pl.read_parquet(file))
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported comparison format: {file.suffix}")
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical_relaxed")


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


def comparison_performance_table(
    settlements: Sequence[Settlement],
    comparisons: pl.DataFrame,
) -> pl.DataFrame:
    """Augment comparison allocations with realised performance metrics."""

    if comparisons.is_empty():
        return comparisons
    join_keys = [
        "event_id",
        "sportsbook",
        "team_or_player",
        "market",
        "scope",
        "side",
        "line",
        "american_odds",
    ]
    settlements_frame = settlements_to_frame(settlements)
    if settlements_frame.is_empty():
        joined = comparisons.with_columns(
            pl.lit(None).alias("stake"),
            pl.lit(None).alias("pnl"),
        )
    else:
        joined = comparisons.join(
            settlements_frame,
            on=join_keys,
            how="left",
            suffix="_settlement",
        )
    safe = joined.with_columns(
        pl.col("allocation_stake").fill_null(0.0),
        pl.col("expected_value").fill_null(0.0),
        pl.col("stake").fill_null(0.0),
        pl.col("pnl").fill_null(0.0),
    )
    with_returns = safe.with_columns(
        pl.when(pl.col("stake") != 0.0)
        .then(pl.col("pnl") / pl.col("stake"))
        .otherwise(0.0)
        .alias("pnl_per_unit"),
    )
    enriched = with_returns.with_columns(
        (pl.col("allocation_stake") * pl.col("expected_value")).alias("expected_pnl"),
        (pl.col("allocation_stake") * pl.col("pnl_per_unit")).alias("realized_pnl"),
    )
    return enriched

