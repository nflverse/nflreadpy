from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nflreadpy.betting.backtesting import (
    BacktestMetrics,
    SportsbookRules,
    closing_line_table,
    export_closing_line_report,
    export_reliability_diagram,
    get_sportsbook_rules,
    load_historical_snapshots,
    reliability_table,
    run_backtest,
    simulate_settlements,
    settlements_to_frame,
)

DATA_PATH = Path(__file__).parent / "data" / "historical_snapshots.csv"


@pytest.fixture(scope="module")
def snapshots() -> pl.DataFrame:
    return load_historical_snapshots(DATA_PATH)


def test_load_historical_snapshots_returns_expected_shape(snapshots: pl.DataFrame) -> None:
    assert snapshots.height == 6
    assert set(snapshots.columns) >= {
        "event_id",
        "sportsbook",
        "market",
        "team_or_player",
        "model_probability",
        "home_team",
        "home_score",
    }


def test_run_backtest_computes_metrics(snapshots: pl.DataFrame) -> None:
    result = run_backtest(snapshots)
    assert isinstance(result, BacktestMetrics)
    assert len(result.settlements) == 6


def test_run_backtest_metrics_are_deterministic(snapshots: pl.DataFrame) -> None:
    result = run_backtest(snapshots)
    assert pytest.approx(result.total_pnl, rel=1e-6) == 2.7307026307026305
    assert pytest.approx(result.average_brier, rel=1e-6) == 0.1971666666666667
    assert pytest.approx(result.average_log_loss, rel=1e-6) == 0.6706081918581456
    assert pytest.approx(result.average_crps, rel=1e-6) == 0.1971666666666667


def test_simulate_settlements_produces_expected_values(
    snapshots: pl.DataFrame,
) -> None:
    settlements = list(simulate_settlements(snapshots))
    moneyline = next(
        item
        for item in settlements
        if item.event_id == "2024-NE-NYJ" and item.market == "moneyline"
    )
    assert moneyline.outcome == "win"
    assert math.isclose(moneyline.actual_outcome, 1.0)
    assert moneyline.pnl == pytest.approx(0.7692307692307693, rel=1e-6)
    assert moneyline.brier == pytest.approx(0.1764, rel=1e-6)
    assert moneyline.log_loss == pytest.approx(0.5447271754416722, rel=1e-6)
    push = next(
        item
        for item in settlements
        if item.event_id == "2024-CHI-GB" and item.market == "spread"
    )
    assert push.outcome == "push"
    assert math.isclose(push.actual_outcome, 0.5)
    assert math.isclose(push.pnl, 0.0)


def test_reliability_diagram_exports_expected_values(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "reliability.csv"
    path = export_reliability_diagram(settlements, output, bins=5)
    table = pl.read_csv(path)
    assert table.height == 1
    row = table.row(0, named=True)
    assert math.isclose(row["bin_lower"], 0.4)
    assert math.isclose(row["bin_upper"], 0.6)
    assert row["count"] == 6
    assert pytest.approx(row["predicted_mean"], rel=1e-6) == 0.53
    assert pytest.approx(row["observed_rate"], rel=1e-6) == 0.75


def test_reliability_table_matches_exported_results(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    table = reliability_table(settlements, bins=5)
    assert table.shape == (1, 5)
    row = table.row(0, named=True)
    assert math.isclose(row["bin_lower"], 0.4)
    assert math.isclose(row["bin_upper"], 0.6)
    assert row["count"] == 6


def test_closing_line_report_contains_odds_and_line_deltas(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "closing.csv"
    path = export_closing_line_report(settlements, output)
    table = pl.read_csv(path)
    assert table.height == 6
    first = table.filter(pl.col("event_id") == "2024-NE-NYJ").sort("market").row(0, named=True)
    assert first["market"] == "moneyline"
    assert pytest.approx(first["odds_delta"], rel=1e-6) == -10.0
    assert pytest.approx(first["implied_probability"], rel=1e-6) == 0.56521739
    assert pytest.approx(first["closing_implied_probability"], rel=1e-6) == 0.58333333
    initial_implied = 13 / 23
    closing_implied = 7 / 12
    expected_model_edge = 0.58 - initial_implied
    expected_closing_edge = 0.58 - closing_implied
    expected_edge_delta = expected_closing_edge - expected_model_edge
    assert first["model_probability"] == pytest.approx(0.58, rel=1e-6)
    assert first["model_edge"] == pytest.approx(expected_model_edge, rel=1e-6)
    assert first["closing_model_edge"] == pytest.approx(expected_closing_edge, rel=1e-6)
    assert first["edge_delta"] == pytest.approx(expected_edge_delta, rel=1e-6)
    spread = table.filter((pl.col("event_id") == "2024-NE-NYJ") & (pl.col("market") == "spread")).row(0, named=True)
    assert pytest.approx(spread["line_delta"], rel=1e-6) == -0.5


def test_closing_line_table_returns_expected_columns(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    table = closing_line_table(settlements)
    assert table.height == 6
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
    assert frame.shape == (6, len(frame.columns))
    assert set(frame.columns) >= {"event_id", "pnl", "brier"}


def test_push_rules_respected_for_testbook(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    push = next(
        item
        for item in settlements
        if item.event_id == "2024-CHI-GB" and item.market == "spread"
    )
    assert push.outcome == "push"
    assert math.isclose(push.actual_outcome, 0.5)
    assert math.isclose(push.pnl, 0.0)


def test_custom_rules_override_default_behaviour(snapshots: pl.DataFrame) -> None:
    overrides = {"testbook": SportsbookRules(name="no_push", push_on_spread=False)}
    result = run_backtest(snapshots, sportsbook_rules=overrides)
    push = next(
        item
        for item in result.settlements
        if item.event_id == "2024-CHI-GB" and item.market == "spread"
    )
    assert push.outcome == "loss"
    assert math.isclose(push.actual_outcome, 0.0)
    assert math.isclose(push.pnl, -push.stake)


def test_get_sportsbook_rules_falls_back_to_default() -> None:
    default_rules = get_sportsbook_rules("unknown")
    assert isinstance(default_rules, SportsbookRules)
    assert default_rules.name == "default"
