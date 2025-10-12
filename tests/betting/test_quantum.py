import math
from dataclasses import dataclass

import pytest

from nflreadpy.betting import (
    PortfolioManager,
    QuantumPortfolioOptimizer,
    SimulatedAnnealingOptimizer,
    QAOAHeuristicOptimizer,
    create_optimizer,
)
from nflreadpy.betting.analytics import Opportunity


@dataclass(slots=True)
class _StubOpportunity:
    event_id: str
    expected_value: float
    kelly_fraction: float

    def to_opportunity(self, *, market: str) -> Opportunity:
        return Opportunity(
            event_id=self.event_id,
            sportsbook="stub",
            book_market_group="grp",
            market=market,
            scope="game",
            entity_type="team",
            team_or_player=self.event_id,
            side=None,
            line=None,
            american_odds=100,
            model_probability=0.55,
            push_probability=0.0,
            implied_probability=0.5,
            expected_value=self.expected_value,
            kelly_fraction=self.kelly_fraction,
            extra={},
        )


@pytest.fixture
def sample_opportunities() -> list[Opportunity]:
    base = [
        _StubOpportunity("A", 0.08, 0.04),
        _StubOpportunity("B", 0.03, 0.02),
        _StubOpportunity("C", 0.12, 0.05),
    ]
    return [stub.to_opportunity(market="market") for stub in base]


def test_create_optimizer_registry_produces_expected_instances() -> None:
    quantum = create_optimizer("quantum", shots=128, temperature=0.4, seed=7)
    annealing = create_optimizer("annealing", steps=256, initial_temperature=1.2, seed=3)
    qaoa = create_optimizer("qaoa", layers=3, gamma=0.7, beta=0.3)

    assert isinstance(quantum, QuantumPortfolioOptimizer)
    assert isinstance(annealing, SimulatedAnnealingOptimizer)
    assert isinstance(qaoa, QAOAHeuristicOptimizer)

    with pytest.raises(ValueError):
        create_optimizer("not-a-solver")


def test_quantum_optimizer_is_deterministic_with_seed(sample_opportunities: list[Opportunity]) -> None:
    solver = QuantumPortfolioOptimizer(shots=256, seed=11, temperature=0.5)
    first = solver.optimise(sample_opportunities, risk_aversion=0.1)
    second = solver.optimise(sample_opportunities, risk_aversion=0.1)
    assert first == second


def test_simulated_annealing_prefers_high_value(sample_opportunities: list[Opportunity]) -> None:
    solver = SimulatedAnnealingOptimizer(steps=2_000, initial_temperature=1.5, cooling_rate=0.995, seed=21)
    ranking = solver.optimise(sample_opportunities, risk_aversion=0.05)
    assert ranking
    top_event = ranking[0][0].event_id
    assert top_event == "C"


def test_portfolio_manager_uses_injected_solver(sample_opportunities: list[Opportunity]) -> None:
    class RecordingOptimizer:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def optimise(self, opportunities, risk_aversion: float = 0.4):
            self.calls.append(risk_aversion)
            weight = 1.0 / max(1, len(opportunities))
            return [(opp, weight) for opp in opportunities]

    recorder = RecordingOptimizer()
    manager = PortfolioManager(1_000.0, optimizer=recorder, default_risk_aversion=0.25)

    first_ranking = manager.rank_opportunities(sample_opportunities)
    second_ranking = manager.rank_opportunities(sample_opportunities, risk_aversion=0.15)

    assert recorder.calls == [0.25, 0.15]
    assert first_ranking
    assert second_ranking
    assert math.isclose(sum(weight for _, weight in first_ranking), 1.0)
    assert manager.last_ranking == second_ranking
