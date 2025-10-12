"""Quantum-inspired and meta-heuristic portfolio allocation optimisers."""

from __future__ import annotations

import dataclasses
import math
import random
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
    runtime_checkable,
)


@runtime_checkable
class OpportunityLike(Protocol):
    """Minimal interface required by the optimisers."""

    expected_value: float
    kelly_fraction: float
    event_id: str
    market: str
    team_or_player: str


OpportunityT = TypeVar("OpportunityT", bound=OpportunityLike)


@runtime_checkable
class PortfolioOptimizer(Protocol[OpportunityT]):
    """Common interface implemented by portfolio solvers."""

    def optimise(
        self,
        opportunities: Sequence[OpportunityT],
        risk_aversion: float = 0.4,
    ) -> List[Tuple[OpportunityT, float]]:
        """Return weighted opportunities ranked by the solver."""


def _score_opportunities(
    opportunities: Sequence[OpportunityT], risk_aversion: float
) -> List[float]:
    scores: List[float] = []
    for opp in opportunities:
        edge = max(0.0, opp.expected_value)
        risk_penalty = risk_aversion * abs(opp.kelly_fraction)
        scores.append(edge - risk_penalty)
    return scores


def _normalise_weights(weights: Iterable[float]) -> List[float]:
    values = [max(0.0, weight) for weight in weights]
    total = sum(values)
    if total <= 0.0:
        if not values:
            return []
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


@dataclasses.dataclass(slots=True)
class QuantumPortfolioOptimizer(PortfolioOptimizer[OpportunityT]):
    """Sample opportunities via amplitude-style weighting."""

    shots: int = 512
    temperature: float = 0.6
    seed: int | None = None

    def optimise(
        self,
        opportunities: Sequence[OpportunityT],
        risk_aversion: float = 0.4,
    ) -> List[Tuple[OpportunityT, float]]:
        if not opportunities:
            return []
        weights = self._amplitudes(opportunities, risk_aversion)
        rng = random.Random(self.seed)
        counts = [0 for _ in opportunities]
        total = 0
        for _ in range(max(0, self.shots)):
            idx = self._sample(weights, rng)
            if idx is None:
                continue
            counts[idx] += 1
            total += 1
        if total == 0:
            return []
        allocations = [
            (opportunities[i], counts[i] / total)
            for i in range(len(opportunities))
            if counts[i] > 0
        ]
        allocations.sort(key=lambda item: item[1], reverse=True)
        return allocations

    def _amplitudes(
        self, opportunities: Sequence[OpportunityT], risk_aversion: float
    ) -> List[float]:
        scores = _score_opportunities(opportunities, risk_aversion)
        if not scores:
            return []
        max_value = max(scores)
        amplitudes: List[float] = []
        total = 0.0
        for value in scores:
            exponent = (value - max_value) / max(1e-6, self.temperature)
            amplitude = math.sqrt(max(1e-9, math.exp(exponent)))
            amplitudes.append(amplitude)
            total += amplitude
        if total <= 0.0:
            if not amplitudes:
                return []
            return [1.0 / len(amplitudes)] * len(amplitudes)
        return [amp / total for amp in amplitudes]

    @staticmethod
    def _sample(weights: Sequence[float], rng: random.Random) -> int | None:
        cumulative = 0.0
        r = rng.random()
        for idx, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return idx
        return None


@dataclasses.dataclass(slots=True)
class SimulatedAnnealingOptimizer(PortfolioOptimizer[OpportunityT]):
    """Explore the search space via simulated annealing."""

    steps: int = 1_024
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    seed: int | None = None

    def optimise(
        self,
        opportunities: Sequence[OpportunityT],
        risk_aversion: float = 0.4,
    ) -> List[Tuple[OpportunityT, float]]:
        if not opportunities or self.steps <= 0:
            return []
        scores = _score_opportunities(opportunities, risk_aversion)
        rng = random.Random(self.seed)
        index = rng.randrange(len(opportunities))
        current_score = scores[index]
        counts = [0 for _ in opportunities]
        temperature = max(1e-6, self.initial_temperature)
        for _ in range(self.steps):
            candidate = rng.randrange(len(opportunities))
            delta = scores[candidate] - current_score
            if delta > 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
                index = candidate
                current_score = scores[index]
            counts[index] += 1
            temperature = max(1e-6, temperature * self.cooling_rate)
        total = sum(counts)
        if total <= 0:
            return []
        allocations = [
            (opportunities[i], counts[i] / total)
            for i in range(len(opportunities))
            if counts[i] > 0
        ]
        allocations.sort(key=lambda item: item[1], reverse=True)
        return allocations


@dataclasses.dataclass(slots=True)
class QAOAHeuristicOptimizer(PortfolioOptimizer[OpportunityT]):
    """Apply QAOA-inspired phase mixing to derive weights."""

    layers: int = 2
    gamma: float = 0.8
    beta: float = 0.45
    seed: int | None = None

    def optimise(
        self,
        opportunities: Sequence[OpportunityT],
        risk_aversion: float = 0.4,
    ) -> List[Tuple[OpportunityT, float]]:
        if not opportunities:
            return []
        scores = _score_opportunities(opportunities, risk_aversion)
        weights = _normalise_weights(scores)
        rng = random.Random(self.seed)
        for layer in range(max(1, self.layers)):
            adjusted: List[float] = []
            for index, base in enumerate(weights):
                phase = self.gamma * scores[index] * (layer + 1)
                adjustment = math.sin(phase) * self.beta
                jitter = (rng.random() - 0.5) * 1e-3
                adjusted.append(max(0.0, base + adjustment + jitter))
            weights = _normalise_weights(adjusted)
        allocations = [
            (opportunity, weight)
            for opportunity, weight in zip(opportunities, weights)
            if weight > 0.0
        ]
        allocations.sort(key=lambda item: item[1], reverse=True)
        return allocations


OptimizerFactory = Callable[..., PortfolioOptimizer[OpportunityT]]


class OptimizerRegistry:
    """Mutable registry mapping solver identifiers to factories."""

    def __init__(self) -> None:
        self._registry: Dict[str, OptimizerFactory] = {}

    def register(self, name: str, factory: OptimizerFactory) -> None:
        key = name.lower()
        self._registry[key] = factory

    def get(self, name: str) -> OptimizerFactory | None:
        return self._registry.get(name.lower())

    def names(self) -> Sequence[str]:
        return tuple(sorted(self._registry))


_REGISTRY = OptimizerRegistry()
_REGISTRY.register("quantum", QuantumPortfolioOptimizer)
_REGISTRY.register("annealing", SimulatedAnnealingOptimizer)
_REGISTRY.register("simulated_annealing", SimulatedAnnealingOptimizer)
_REGISTRY.register("qaoa", QAOAHeuristicOptimizer)
_REGISTRY.register("qaoa_heuristic", QAOAHeuristicOptimizer)


def register_optimizer(name: str, factory: OptimizerFactory) -> None:
    """Register ``factory`` under ``name`` in the optimiser registry."""

    _REGISTRY.register(name, factory)


def optimizer_registry() -> Mapping[str, OptimizerFactory]:
    """Return a read-only view of the optimiser registry."""

    return dict(_REGISTRY._registry)


def create_optimizer(
    name: str,
    **parameters: float | int | None,
) -> PortfolioOptimizer[OpportunityT]:
    """Create an optimiser instance given its registered ``name``."""

    factory = _REGISTRY.get(name)
    if not factory:
        available = ", ".join(_REGISTRY.names()) or "<none>"
        raise ValueError(f"Unknown optimiser '{name}'. Available: {available}")

    # Filter parameters to those accepted by the target factory.
    try:
        allowed = {field.name for field in dataclasses.fields(factory)}  # type: ignore[arg-type]
    except TypeError:
        allowed = set()
    filtered = {
        key: value
        for key, value in parameters.items()
        if value is not None and (not allowed or key in allowed)
    }
    return factory(**filtered)


__all__ = [
    "OpportunityLike",
    "PortfolioOptimizer",
    "QAOAHeuristicOptimizer",
    "QuantumPortfolioOptimizer",
    "SimulatedAnnealingOptimizer",
    "create_optimizer",
    "optimizer_registry",
    "register_optimizer",
]

