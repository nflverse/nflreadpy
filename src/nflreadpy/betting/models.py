"""Simulation and probability models for the betting toolkit."""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class TeamRating:
    """Represents a team's strength profile used in simulations."""

    team: str
    offensive_rating: float
    defensive_rating: float


@dataclasses.dataclass(slots=True)
class GameSimulationConfig:
    """Configuration for the Monte Carlo engine."""

    iterations: int = 10_000
    seed: int | None = None


@dataclasses.dataclass(slots=True)
class ProbabilityTriple:
    """Container for win/push/loss probabilities."""

    win: float
    push: float = 0.0

    @property
    def loss(self) -> float:
        return max(0.0, 1.0 - self.win - self.push)


def _distribution_moments(distribution: Mapping[int, int]) -> tuple[float, float]:
    total = sum(distribution.values())
    if total == 0:
        return 0.0, 1.0
    mean = sum(value * count for value, count in distribution.items()) / total
    mean_sq = sum((value**2) * count for value, count in distribution.items()) / total
    variance = max(1e-6, mean_sq - mean**2)
    return mean, variance


def _normal_cdf(x: float, mean: float, stdev: float) -> float:
    if stdev <= 1e-9:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (stdev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _poisson_cdf(k: int, lam: float) -> float:
    cumulative = 0.0
    term = math.exp(-lam)
    cumulative += term
    for i in range(1, k + 1):
        term *= lam / i
        cumulative += term
    return min(cumulative, 1.0)


@dataclasses.dataclass(slots=True)
class SimulationResult:
    event_id: str
    home_team: str
    away_team: str
    iterations: int
    home_win_probability: float
    away_win_probability: float
    expected_margin: float
    expected_total: float
    margin_distribution: Mapping[int, int]
    total_distribution: Mapping[int, int]
    home_score_distribution: Mapping[int, int]
    away_score_distribution: Mapping[int, int]

    def moneyline_probability(self, team: str) -> float:
        if team == self.home_team:
            return self.home_win_probability
        if team == self.away_team:
            return self.away_win_probability
        raise KeyError(f"Team {team} not part of simulation {self.event_id}")

    def tie_probability(self) -> float:
        return self.margin_distribution.get(0, 0) / max(1, self.iterations)

    def spread_probability(self, team: str, line: float, scope: str = "game") -> ProbabilityTriple:
        if scope == "game":
            return self._spread_probability_game(team, line)
        return self._spread_probability_scaled(team, line, scope)

    def total_probability(self, side: str, line: float, scope: str = "game") -> ProbabilityTriple:
        if scope == "game":
            return self._total_probability_game(side, line)
        return self._total_probability_scaled(side, line, scope)

    def team_total_probability(self, team: str, side: str, line: float, scope: str = "game") -> ProbabilityTriple:
        if scope == "game":
            return self._team_total_probability_game(team, side, line)
        return self._team_total_probability_scaled(team, side, line, scope)

    def _spread_probability_game(self, team: str, line: float) -> ProbabilityTriple:
        wins = 0
        pushes = 0
        for margin, count in self.margin_distribution.items():
            outcome = self._spread_outcome_value(team, line, margin)
            if outcome > 0:
                wins += count
            elif outcome == 0:
                pushes += count
        total = max(1, self.iterations)
        return ProbabilityTriple(wins / total, pushes / total)

    def _spread_probability_scaled(self, team: str, line: float, scope: str) -> ProbabilityTriple:
        mean, variance = _distribution_moments(self.margin_distribution)
        factor = _scope_factor(scope)
        adj_mean = mean * factor
        adj_stdev = math.sqrt(variance * max(factor, 1e-6))
        if team == self.away_team:
            adj_mean = -adj_mean
            line = -line
        win = 1.0 - _normal_cdf(line, adj_mean, adj_stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def _total_probability_game(self, side: str, line: float) -> ProbabilityTriple:
        wins = 0
        pushes = 0
        for total, count in self.total_distribution.items():
            if side == "over":
                if total > line:
                    wins += count
                elif total == line:
                    pushes += count
            else:
                if total < line:
                    wins += count
                elif total == line:
                    pushes += count
        total_count = max(1, self.iterations)
        return ProbabilityTriple(wins / total_count, pushes / total_count)

    def _total_probability_scaled(self, side: str, line: float, scope: str) -> ProbabilityTriple:
        mean, variance = _distribution_moments(self.total_distribution)
        factor = _scope_factor(scope)
        adj_mean = mean * factor
        adj_stdev = math.sqrt(variance * max(factor, 1e-6))
        if side == "over":
            win = 1.0 - _normal_cdf(line, adj_mean, adj_stdev)
        else:
            win = _normal_cdf(line, adj_mean, adj_stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def _team_total_probability_game(self, team: str, side: str, line: float) -> ProbabilityTriple:
        distribution = (
            self.home_score_distribution
            if team == self.home_team
            else self.away_score_distribution
        )
        wins = 0
        pushes = 0
        for score, count in distribution.items():
            if side == "over":
                if score > line:
                    wins += count
                elif score == line:
                    pushes += count
            else:
                if score < line:
                    wins += count
                elif score == line:
                    pushes += count
        total = max(1, self.iterations)
        return ProbabilityTriple(wins / total, pushes / total)

    def _team_total_probability_scaled(self, team: str, side: str, line: float, scope: str) -> ProbabilityTriple:
        distribution = (
            self.home_score_distribution
            if team == self.home_team
            else self.away_score_distribution
        )
        mean, variance = _distribution_moments(distribution)
        factor = _scope_factor(scope)
        adj_mean = mean * factor
        adj_stdev = math.sqrt(variance * max(factor, 1e-6))
        if side == "over":
            win = 1.0 - _normal_cdf(line, adj_mean, adj_stdev)
        else:
            win = _normal_cdf(line, adj_mean, adj_stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def _spread_outcome_value(self, team: str, line: float, margin: int) -> float:
        if team == self.home_team:
            return margin + line
        if team == self.away_team:
            return -margin + line
        raise KeyError(f"Team {team} not part of simulation {self.event_id}")


class MonteCarloEngine:
    """Monte Carlo engine producing distributions for multiple markets."""

    def __init__(
        self, ratings: Mapping[str, TeamRating], config: GameSimulationConfig | None = None
    ) -> None:
        self.ratings = dict(ratings)
        self.config = config or GameSimulationConfig()
        self._rng = random.Random(self.config.seed)

    def _scoring_rate(self, offense: TeamRating, defense: TeamRating) -> float:
        baseline = 21.5
        modifier = offense.offensive_rating - defense.defensive_rating
        return max(3.0, baseline * math.exp(modifier / 10.0))

    def simulate_game(self, event_id: str, home_team: str, away_team: str) -> SimulationResult:
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        home_rate = self._scoring_rate(home_rating, away_rating)
        away_rate = self._scoring_rate(away_rating, home_rating)
        margin_distribution: MutableMapping[int, int] = collections.Counter()
        total_distribution: MutableMapping[int, int] = collections.Counter()
        home_distribution: MutableMapping[int, int] = collections.Counter()
        away_distribution: MutableMapping[int, int] = collections.Counter()
        home_wins = 0
        away_wins = 0
        margin_total = 0.0
        total_points = 0.0
        for _ in range(self.config.iterations):
            home_score = self._rng.poisson(home_rate)
            away_score = self._rng.poisson(away_rate)
            margin = home_score - away_score
            total = home_score + away_score
            margin_distribution[margin] += 1
            total_distribution[total] += 1
            home_distribution[home_score] += 1
            away_distribution[away_score] += 1
            margin_total += margin
            total_points += total
            if margin > 0:
                home_wins += 1
            elif margin < 0:
                away_wins += 1
            else:  # tie -> overtime coin flip proxy
                if self._rng.random() < 0.5:
                    home_wins += 1
                else:
                    away_wins += 1
        iterations = self.config.iterations
        result = SimulationResult(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            iterations=iterations,
            home_win_probability=home_wins / iterations,
            away_win_probability=away_wins / iterations,
            expected_margin=margin_total / iterations,
            expected_total=total_points / iterations,
            margin_distribution=dict(margin_distribution),
            total_distribution=dict(total_distribution),
            home_score_distribution=dict(home_distribution),
            away_score_distribution=dict(away_distribution),
        )
        logger.debug("Simulated %s vs %s -> %s", home_team, away_team, result)
        return result

    def simulate_many(
        self, fixtures: Iterable[tuple[str, str, str]]
    ) -> List[SimulationResult]:
        return [
            self.simulate_game(event_id, home_team, away_team)
            for event_id, home_team, away_team in fixtures
        ]


@dataclasses.dataclass(slots=True)
class PlayerProjection:
    player: str
    market: str
    mean: float
    stdev: float
    distribution: str = "normal"


class PlayerPropForecaster:
    """Lightweight probabilistic model for player propositions."""

    def __init__(self, projections: Iterable[PlayerProjection] | None = None) -> None:
        self._projections: Dict[tuple[str, str], PlayerProjection] = {}
        if projections:
            for projection in projections:
                key = (projection.player, projection.market)
                self._projections[key] = projection

    def register_projection(self, projection: PlayerProjection) -> None:
        self._projections[(projection.player, projection.market)] = projection

    def probability(
        self,
        player: str,
        market: str,
        side: str,
        line: float | None,
        scope: str,
        extra: Mapping[str, object] | None = None,
    ) -> ProbabilityTriple:
        projection = self._resolve_projection(player, market, extra)
        if projection is None or line is None:
            return ProbabilityTriple(0.5)
        factor = _scope_factor(scope)
        if projection.distribution == "bernoulli":
            prob = min(1.0, max(0.0, projection.mean * factor))
            if side == "yes":
                return ProbabilityTriple(prob)
            if side == "no":
                return ProbabilityTriple(1.0 - prob)
        if projection.distribution == "poisson":
            lam = max(1e-6, projection.mean * factor)
            threshold = int(math.floor(line))
            if side == "over":
                win = 1.0 - _poisson_cdf(threshold, lam)
            else:
                win = _poisson_cdf(threshold, lam)
            return ProbabilityTriple(max(0.0, min(1.0, win)))
        mean = projection.mean * factor
        stdev = max(1.0, projection.stdev * math.sqrt(max(factor, 1e-6)))
        if side == "over":
            win = 1.0 - _normal_cdf(line, mean, stdev)
        else:
            win = _normal_cdf(line, mean, stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def projection_stats(
        self,
        player: str,
        market: str,
        scope: str,
        extra: Mapping[str, object] | None = None,
    ) -> Tuple[float, float]:
        projection = self._resolve_projection(player, market, extra)
        if projection is None:
            return 0.0, 1.0
        factor = _scope_factor(scope)
        mean = projection.mean * factor
        stdev = max(1.0, projection.stdev * math.sqrt(max(factor, 1e-6)))
        if projection.distribution == "bernoulli":
            p = min(1.0, max(0.0, projection.mean * factor))
            mean = p
            stdev = math.sqrt(max(1e-6, p * (1.0 - p)))
        elif projection.distribution == "poisson":
            lam = max(1e-6, projection.mean * factor)
            mean = lam
            stdev = math.sqrt(lam)
        return mean, stdev

    def _resolve_projection(
        self, player: str, market: str, extra: Mapping[str, object] | None
    ) -> PlayerProjection | None:
        base_market = market
        if market.endswith("_alt"):
            base_market = market[:-4]
        direct = self._projections.get((player, market))
        if direct:
            return direct
        base = self._projections.get((player, base_market))
        if base:
            return base
        if extra and "projection_mean" in extra:
            mean = float(extra["projection_mean"])
            stdev = float(extra.get("projection_stdev", max(5.0, mean * 0.2)))
            distribution = str(extra.get("projection_distribution", "normal"))
            projection = PlayerProjection(
                player=player,
                market=base_market,
                mean=mean,
                stdev=stdev,
                distribution=distribution,
            )
            self.register_projection(projection)
            return projection
        return None


def _scope_factor(scope: str) -> float:
    scope = scope.lower()
    if scope == "game":
        return 1.0
    if scope in {"1h", "first_half"}:
        return 0.52
    if scope in {"2h", "second_half"}:
        return 0.48
    if scope in {"1q", "first_quarter"}:
        return 0.27
    if scope in {"2q", "second_quarter"}:
        return 0.23
    if scope in {"3q", "third_quarter"}:
        return 0.26
    if scope in {"4q", "fourth_quarter"}:
        return 0.24
    return 1.0


# random.Random lacks poisson until py3.12; provide fallback for <=3.11
if not hasattr(random.Random, "poisson"):

    def _poisson(self: random.Random, lam: float) -> int:  # type: ignore[override]
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.random()
        return k - 1

    random.Random.poisson = _poisson  # type: ignore[attr-defined]

