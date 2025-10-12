"""Quantum-inspired portfolio allocation heuristics."""

from __future__ import annotations

import dataclasses
import math
import random
from typing import List, Sequence, Tuple

from .analytics import Opportunity


@dataclasses.dataclass(slots=True)
class QuantumPortfolioOptimizer:
    """Sample opportunities via amplitude-style weighting."""

    shots: int = 512
    temperature: float = 0.6
    seed: int | None = None

    def optimise(
        self,
        opportunities: Sequence[Opportunity],
        risk_aversion: float = 0.4,
    ) -> List[Tuple[Opportunity, float]]:
        if not opportunities:
            return []
        weights = self._amplitudes(opportunities, risk_aversion)
        rng = random.Random(self.seed)
        counts = [0 for _ in opportunities]
        total = 0
        for _ in range(self.shots):
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
        self, opportunities: Sequence[Opportunity], risk_aversion: float
    ) -> List[float]:
        values: List[float] = []
        max_value = -math.inf
        for opp in opportunities:
            edge = max(0.0, opp.expected_value)
            risk_penalty = risk_aversion * abs(opp.kelly_fraction)
            score = edge - risk_penalty
            max_value = max(max_value, score)
            values.append(score)
        if not values:
            return []
        # softmax-like transform to compute amplitudes
        amplitudes: List[float] = []
        total = 0.0
        for value in values:
            exponent = (value - max_value) / max(1e-6, self.temperature)
            amplitude = math.sqrt(max(1e-9, math.exp(exponent)))
            amplitudes.append(amplitude)
            total += amplitude
        if total <= 0.0:
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

