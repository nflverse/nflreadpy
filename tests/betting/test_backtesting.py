from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nflreadpy.betting.backtesting import (
    BacktestMetrics,
    export_closing_line_report,
    export_reliability_diagram,
    load_historical_snapshots,
    run_backtest,
    settlements_to_frame,
)

DATA_PATH = Path(__file__).parent / "data" / "historical_snapshots.csv"


@pytest.fixture(scope="module")
def snapshots() -> pl.DataFrame:
    return load_historical_snapshots(DATA_PATH)


def test_load_historical_snapshots_returns_expected_shape(snapshots: pl.DataFrame) -> None:
    assert snapshots.height == 5
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
    assert pytest.approx(result.total_pnl, rel=1e-6) == 2.73070262
    assert pytest.approx(result.average_brier, rel=1e-6) == 0.23588
    assert pytest.approx(result.average_log_loss, rel=1e-6) == 0.6646498
    assert pytest.approx(result.average_crps, rel=1e-6) == 0.23588
    assert len(result.settlements) == 5


def test_reliability_diagram_exports_expected_values(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "reliability.csv"
    path = export_reliability_diagram(settlements, output, bins=5)
    table = pl.read_csv(path)
    assert table.height == 1
    row = table.row(0, named=True)
    assert math.isclose(row["bin_lower"], 0.4)
    assert math.isclose(row["bin_upper"], 0.6)
    assert row["count"] == 5
    assert pytest.approx(row["predicted_mean"], rel=1e-6) == 0.524
    assert pytest.approx(row["observed_rate"], rel=1e-6) == 0.8


def test_closing_line_report_contains_odds_and_line_deltas(tmp_path: Path, snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    output = tmp_path / "closing.csv"
    path = export_closing_line_report(settlements, output)
    table = pl.read_csv(path)
    assert table.height == 5
    first = table.filter(pl.col("event_id") == "2024-NE-NYJ").sort("market").row(0, named=True)
    assert first["market"] == "moneyline"
    assert pytest.approx(first["odds_delta"], rel=1e-6) == -10.0
    assert pytest.approx(first["implied_probability"], rel=1e-6) == 0.56521739
    assert pytest.approx(first["closing_implied_probability"], rel=1e-6) == 0.58333333
    spread = table.filter((pl.col("event_id") == "2024-NE-NYJ") & (pl.col("market") == "spread")).row(0, named=True)
    assert pytest.approx(spread["line_delta"], rel=1e-6) == -0.5


def test_settlements_to_frame_round_trips_data(snapshots: pl.DataFrame) -> None:
    settlements = run_backtest(snapshots).settlements
    frame = settlements_to_frame(settlements)
    assert frame.shape == (5, len(frame.columns))
    assert set(frame.columns) >= {"event_id", "pnl", "brier"}
