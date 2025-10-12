from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from nflreadpy.betting.analytics import (
    Opportunity,
    compare_optimizers,
    persist_optimizer_comparison,
)
from nflreadpy.betting.backtesting import (
    Settlement,
    comparison_performance_table,
    load_comparison_history,
)
from nflreadpy.betting.quantum import QuantumPortfolioOptimizer
from nflreadpy.betting.utils import (
    american_to_profit_multiplier,
    implied_probability_from_american,
)


def _make_opportunity(
    event_id: str,
    american_odds: int,
    model_probability: float,
    *,
    push_probability: float = 0.0,
    kelly_fraction: float = 0.1,
) -> Opportunity:
    implied = implied_probability_from_american(american_odds)
    multiplier = american_to_profit_multiplier(american_odds)
    expected_value = model_probability * multiplier - (
        1.0 - model_probability - push_probability
    )
    return Opportunity(
        event_id=event_id,
        sportsbook="testbook",
        book_market_group="main",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player=f"{event_id}-team",
        side=None,
        line=None,
        american_odds=american_odds,
        model_probability=model_probability,
        push_probability=push_probability,
        implied_probability=implied,
        expected_value=expected_value,
        kelly_fraction=kelly_fraction,
        extra={},
    )


def _sample_opportunities() -> list[Opportunity]:
    return [
        _make_opportunity("game-1", -110, 0.56, push_probability=0.02, kelly_fraction=0.12),
        _make_opportunity("game-2", +150, 0.42, push_probability=0.01, kelly_fraction=0.18),
        _make_opportunity("game-3", +200, 0.38, push_probability=0.0, kelly_fraction=0.22),
    ]


def test_compare_optimizers_deterministic() -> None:
    opportunities = _sample_opportunities()
    optimizer = QuantumPortfolioOptimizer(shots=64, temperature=0.7, seed=42)
    result_a = compare_optimizers(
        opportunities,
        bankroll=750.0,
        fractional_kelly=0.6,
        max_risk_per_bet=0.05,
        quantum_optimizer=optimizer,
        risk_aversion=0.3,
        metadata={"tag": "first"},
        run_id="test-a",
    )
    result_b = compare_optimizers(
        opportunities,
        bankroll=750.0,
        fractional_kelly=0.6,
        max_risk_per_bet=0.05,
        quantum_optimizer=optimizer,
        risk_aversion=0.3,
        run_id="test-b",
    )
    assert pytest.approx(result_a.classical.total_stake) == result_b.classical.total_stake
    assert pytest.approx(result_a.quantum.total_stake) == result_b.quantum.total_stake
    assert pytest.approx(result_a.overlap_fraction) == result_b.overlap_fraction
    assert pytest.approx(result_a.expected_value_difference) == result_b.expected_value_difference
    stakes_a = [position.stake for position in result_a.quantum.positions]
    stakes_b = [position.stake for position in result_b.quantum.positions]
    assert stakes_a == pytest.approx(stakes_b)


def test_persist_and_performance_pipeline(tmp_path: Path) -> None:
    opportunities = _sample_opportunities()
    comparison = compare_optimizers(
        opportunities,
        bankroll=500.0,
        fractional_kelly=0.5,
        max_risk_per_bet=0.05,
        quantum_optimizer=QuantumPortfolioOptimizer(shots=32, seed=7),
        risk_aversion=0.25,
        run_id="persist-test",
    )
    summary_path, allocations_path = persist_optimizer_comparison(comparison, tmp_path)
    summary = json.loads(summary_path.read_text())
    assert summary["comparison_id"] == comparison.run_id
    allocations = pl.read_csv(allocations_path)
    assert set(allocations.get_column("optimizer").to_list()) == {"classical", "quantum"}

    history = load_comparison_history(allocations_path)
    assert history.shape == allocations.shape

    unit_profit = american_to_profit_multiplier(-110)
    settlement = Settlement(
        event_id="game-1",
        sportsbook="testbook",
        team_or_player="game-1-team",
        market="moneyline",
        scope="game",
        snapshot_type=None,
        side=None,
        line=None,
        american_odds=-110,
        stake=1.0,
        model_probability=0.56,
        outcome="win",
        actual_outcome=1.0,
        pnl=unit_profit,
        brier=(0.56 - 1.0) ** 2,
        log_loss=-math.log(0.56),
        crps=(0.56 - 1.0) ** 2,
        closing_american_odds=None,
        closing_line=None,
    )
    performance = comparison_performance_table([settlement], history)
    classical_rows = performance.filter(pl.col("optimizer") == "classical")
    assert "expected_pnl" in classical_rows.columns
    residual = classical_rows.select(
        (pl.col("expected_pnl") - pl.col("allocation_stake") * pl.col("expected_value")).abs()
    ).to_series()
    assert residual.max() < 1e-9
    realised = classical_rows.filter(pl.col("event_id") == "game-1").get_column("realized_pnl")
    if realised.len() > 0:
        assert realised.max() >= 0.0
