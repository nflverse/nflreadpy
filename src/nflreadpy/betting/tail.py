"""Monotone tail probability smoothing for ladder markets."""

from __future__ import annotations

import dataclasses
from typing import List, Literal, Sequence

DirectionLiteral = Literal["increasing", "decreasing"]


@dataclasses.dataclass(frozen=True)
class LadderPoint:
    """Represents a ladder line estimate prior to smoothing."""

    line: float
    win_probability: float
    push_probability: float = 0.0
    weight: float = 1.0


def fit_monotone_envelope(
    points: Sequence[LadderPoint],
    *,
    direction: DirectionLiteral,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[float]:
    """Return win probabilities constrained to be monotone in ``line``.

    Parameters
    ----------
    points:
        Ordered ladder points (ascending by line) to be smoothed.
    direction:
        Whether probabilities should be non-decreasing (``"increasing"``)
        or non-increasing (``"decreasing"``) as the line grows.
    alpha, beta:
        Beta prior parameters used to shrink extremely small sample ladders
        towards a neutral win probability.
    """

    if not points:
        return []
    if direction not in {"increasing", "decreasing"}:
        raise ValueError(f"Unsupported direction: {direction}")
    smoothed: List[float] = []
    effective_values: List[float] = []
    weights: List[float] = []
    for point in points:
        win = max(0.0, min(1.0, float(point.win_probability)))
        push = max(0.0, min(1.0, float(point.push_probability)))
        weight = max(1e-6, float(point.weight))
        non_push_weight = weight * max(1e-6, 1.0 - push)
        successes = win * non_push_weight
        failures = max(0.0, non_push_weight - successes)
        posterior = (successes + alpha) / (successes + failures + alpha + beta)
        maximum = 1.0 - push
        effective_values.append(min(maximum, max(0.0, posterior)))
        weights.append(non_push_weight)
    if direction == "decreasing":
        effective_values = list(reversed(effective_values))
        weights = list(reversed(weights))
        pav = _pool_adjacent_violators(effective_values, weights)
        smoothed = list(reversed(pav))
    else:
        smoothed = _pool_adjacent_violators(effective_values, weights)
    return smoothed


def _pool_adjacent_violators(values: Sequence[float], weights: Sequence[float]) -> List[float]:
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    averages = list(values)
    block_weights = list(weights)
    block_sizes = [1] * len(values)
    index = 0
    while index < len(averages) - 1:
        if averages[index] > averages[index + 1] + 1e-12:
            total_weight = block_weights[index] + block_weights[index + 1]
            if total_weight <= 0.0:
                total_weight = 1e-6
            merged_value = (
                (averages[index] * block_weights[index])
                + (averages[index + 1] * block_weights[index + 1])
            ) / total_weight
            averages[index] = merged_value
            block_weights[index] = total_weight
            block_sizes[index] += block_sizes[index + 1]
            del averages[index + 1]
            del block_weights[index + 1]
            del block_sizes[index + 1]
            if index > 0:
                index -= 1
        else:
            index += 1
    smoothed: List[float] = []
    for value, size in zip(averages, block_sizes):
        smoothed.extend([value] * size)
    return smoothed


__all__ = ["LadderPoint", "fit_monotone_envelope"]
