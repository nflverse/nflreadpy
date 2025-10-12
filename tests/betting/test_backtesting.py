from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from nflreadpy.betting.backtesting import (
    BacktestArtifacts,
    BacktestMetrics,
    QuantumComparisonArtifacts,
    QuantumScenarioComparison,
    SportsbookRules,
    closing_line_table,
    compare_quantum_backtest,
    export_closing_line_report,
    export_reliability_diagram,
    get_sportsbook_rules,
    load_historical_snapshots,
    persist_backtest_reports,
    persist_quantum_comparison,
    reliability_table,
    run_backtest,
    settlements_to_frame,
    simulate_settlements,
)

DATA_PATH = Path(__file__).parent / "data" / "historical_snapshots.csv"


@pytest.fixture(scope="module")
def snapshots() -> pl.DataFrame:
    return load_historical_snapshots(DATA_PATH)


def test_load_historical_snapshots_returns_expected_shape(snapshots: pl.DataFrame) -> None:
    assert snapshots.height == 4
    assert set(snapshots.columns) >= {
        "event_id",
        "sportsbook",
        "market",
        "team_or_player",
        "model_probability",
        "snapshot_type",
        "home_team",
        "home_score",
    }


def test_run_backtest_computes_metrics(snapshots: pl.DataFrame) -> None:
    result = run_backtest(snapshots)
    assert isinstance(result, BacktestMetrics)
    assert len(result.settlements) == 4


def test_run_backtest_metrics_are_deterministic(snapshots: pl.DataFrame) -> None:
    result = run_backtest(snapshots)
    assert pytest.approx(result.total_pnl, rel=1e-6) == 2.885714285714286
    assert pytest.approx(result.average_brier, rel=1e-6) == 0.16625
    assert pytest.approx(result.average_log_loss, rel=1e-6) == 0.6500793753248321
    assert pytest.approx(result.average_crps, rel=1e-6) == 0.16625


def test_simulate_settlements_produces_expected_values(
    snapshots: pl.DataFrame,
) -> None:
    settlements = list(simulate_settlements(snapshots))
    moneyline = next(
        item
        for item in settlements
        if item.event_id == "2024-SEA-SF" and item.market == "moneyline"
    )
    assert moneyline.outcome == "win"
    assert math.isclose(moneyline.actual_outcome, 1.0)
    assert moneyline.pnl == pytest.approx(1.1, rel=1e-6)
    assert moneyline.brier == pytest.approx(0.16, rel=1e-6)
    assert moneyline.log_loss == pytest.approx(0.5108256237659907, rel=1e-6)
    assert moneyline.snapshot_type == "open"
    push = next(
        item
        for item in settlements
        if item.event_id == "2024-LAR-ARI" and item.market == "spread"
    )
    assert push.outcome == "push"
    assert math.isclose(push.actual_outcome, 0.5)
    assert math.isclose(push.pnl, 0.0)
    assert math.isclose(push.stake, 2.0)


def test_reliability_diagram_exports_expected_values(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "reliability.csv"
    path = export_reliability_diagram(settlements, output, bins=10)
    table = pl.read_csv(path).sort("bin_lower")
    assert table.height == 3

    first = table.row(0, named=True)
    assert math.isclose(first["bin_lower"], 0.4)
    assert math.isclose(first["bin_upper"], 0.5)
    assert first["count"] == 1
    assert pytest.approx(first["predicted_mean"], rel=1e-6) == 0.45
    assert pytest.approx(first["observed_rate"], rel=1e-6) == 1.0

    second = table.row(1, named=True)
    assert math.isclose(second["bin_lower"], 0.5)
    assert math.isclose(second["bin_upper"], 0.6)
    assert second["count"] == 2
    assert pytest.approx(second["predicted_mean"], rel=1e-6) == 0.525
    assert pytest.approx(second["observed_rate"], rel=1e-6) == 0.75

    third = table.row(2, named=True)
    assert math.isclose(third["bin_lower"], 0.6)
    assert math.isclose(third["bin_upper"], 0.7)
    assert third["count"] == 1
    assert pytest.approx(third["predicted_mean"], rel=1e-6) == 0.6
    assert pytest.approx(third["observed_rate"], rel=1e-6) == 1.0


def test_reliability_table_matches_exported_results(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    table = reliability_table(settlements, bins=10).sort("bin_lower")
    assert table.shape == (3, 5)


def test_closing_line_report_contains_odds_and_line_deltas(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "closing.csv"
    path = export_closing_line_report(settlements, output)
    table = pl.read_csv(path)
    assert table.height == 4

    moneyline = (
        table
        .filter(pl.col("event_id") == "2024-SEA-SF")
        .filter(pl.col("market") == "moneyline")
        .row(0, named=True)
    )
    assert pytest.approx(moneyline["odds_delta"], rel=1e-6) == -10.0
    initial_implied = 100 / (110 + 100)
    closing_implied = 100 / (100 + 100)
    assert pytest.approx(moneyline["implied_probability"], rel=1e-6) == initial_implied
    assert pytest.approx(moneyline["closing_implied_probability"], rel=1e-6) == closing_implied
    expected_model_edge = 0.6 - initial_implied
    expected_closing_edge = 0.6 - closing_implied
    expected_edge_delta = expected_closing_edge - expected_model_edge
    assert moneyline["model_probability"] == pytest.approx(0.6, rel=1e-6)
    assert moneyline["model_edge"] == pytest.approx(expected_model_edge, rel=1e-6)
    assert moneyline["closing_model_edge"] == pytest.approx(expected_closing_edge, rel=1e-6)
    assert moneyline["edge_delta"] == pytest.approx(expected_edge_delta, rel=1e-6)

    spread = (
        table
        .filter((pl.col("event_id") == "2024-NO-ATL") & (pl.col("market") == "spread"))
        .row(0, named=True)
    )
    assert pytest.approx(spread["line_delta"], rel=1e-6) == -0.5


def test_closing_line_table_returns_expected_columns(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    table = closing_line_table(settlements)
    assert table.height == 4
    assert {
        "odds_delta",
        "implied_delta",
        "line_delta",
        "model_edge",
        "closing_model_edge",
        "edge_delta",
    }.issubset(table.columns)


def test_settlements_to_frame_round_trips_data(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    frame = settlements_to_frame(settlements)
    assert frame.shape == (4, len(frame.columns))
    assert set(frame.columns) >= {"event_id", "pnl", "brier", "snapshot_type"}


def test_push_rules_respected_for_default_rules(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    push = next(
        item
        for item in settlements
        if item.event_id == "2024-LAR-ARI" and item.market == "spread"
    )
    assert push.outcome == "push"
    assert math.isclose(push.actual_outcome, 0.5)
    assert math.isclose(push.pnl, 0.0)


def test_persist_backtest_reports_writes_expected_artifacts(
    tmp_path: Path, snapshots: pl.DataFrame
) -> None:
    metrics = run_backtest(snapshots)
    artefacts = persist_backtest_reports(
        metrics,
        tmp_path,
        bins=5,
        reliability_filename="calibration.csv",
        closing_line_filename="closing_efficiency.csv",
    )
    assert isinstance(artefacts, BacktestArtifacts)
    assert artefacts.reliability_path.name == "calibration.csv"
    assert artefacts.closing_line_path.name == "closing_efficiency.csv"
    assert artefacts.reliability_path.exists()
    assert artefacts.closing_line_path.exists()

    reliability = pl.read_csv(artefacts.reliability_path)
    closing = pl.read_csv(artefacts.closing_line_path)

    expected_reliability = reliability_table(metrics.settlements, bins=5)
    expected_closing = closing_line_table(metrics.settlements)

    plt.assert_frame_equal(reliability, expected_reliability)
    plt.assert_frame_equal(closing, expected_closing)


def test_custom_rules_override_default_behaviour(snapshots: pl.DataFrame) -> None:
    overrides = {"nopushbook": SportsbookRules(name="no_push", push_on_spread=False)}
    result = run_backtest(snapshots, sportsbook_rules=overrides)
    push = next(
        item
        for item in result.settlements
        if item.event_id == "2024-LAR-ARI" and item.market == "spread"
    )
    assert push.outcome == "loss"
    assert math.isclose(push.actual_outcome, 0.0)
    assert math.isclose(push.pnl, -push.stake)


def test_get_sportsbook_rules_falls_back_to_default() -> None:
    default_rules = get_sportsbook_rules("unknown")
    assert isinstance(default_rules, SportsbookRules)
    assert default_rules.name == "default"


def test_compare_quantum_backtest_returns_comparison(
    snapshots: pl.DataFrame,
) -> None:
    comparison = compare_quantum_backtest(
        snapshots,
        bankroll=100.0,
        max_risk_per_bet=0.25,
        max_event_exposure=1.0,
        portfolio_fraction=1.0,
        optimizer_shots=256,
        optimizer_seed=7,
        bootstrap_iterations=512,
        bootstrap_seed=11,
    )
    assert isinstance(comparison, QuantumScenarioComparison)
    assert len(comparison.profit_differences) == snapshots.height
    assert comparison.baseline.name == "baseline"
    assert comparison.quantum.name == "quantum"
    assert comparison.bootstrap_p_value is not None
    assert comparison.t_statistic is not None
    assert comparison.baseline.expected_value == pytest.approx(6.1518181818)
    assert comparison.delta_expected_value == pytest.approx(0.0)
    assert comparison.baseline.per_event_profits[0] == pytest.approx(26.0)


def test_persist_quantum_comparison_creates_expected_files(
    tmp_path: Path, snapshots: pl.DataFrame
) -> None:
    comparison = compare_quantum_backtest(
        snapshots,
        bankroll=100.0,
        max_risk_per_bet=0.25,
        max_event_exposure=1.0,
        portfolio_fraction=1.0,
        optimizer_shots=128,
        optimizer_seed=3,
        bootstrap_iterations=128,
        bootstrap_seed=5,
    )
    artefacts = persist_quantum_comparison(comparison, tmp_path / "reports")
    assert isinstance(artefacts, QuantumComparisonArtifacts)
    summary = pl.read_csv(artefacts.summary_path)
    assert summary.height == 3
    assert {"scenario", "expected_value", "hit_rate"}.issubset(summary.columns)
    distribution = pl.read_csv(artefacts.distribution_path)
    assert set(distribution["scenario"].unique().to_list()) == {
        "baseline",
        "quantum",
        "difference",
    }
    significance = json.loads(artefacts.significance_path.read_text())
    assert "t_statistic" in significance
    assert "bootstrap_p_value" in significance
    assert significance["delta_expected_value"] == pytest.approx(0.0)
