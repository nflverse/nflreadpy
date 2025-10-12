"""Simulation and probability models for the betting toolkit."""

from __future__ import annotations

import collections
import dataclasses
import importlib
import itertools
import logging
import math
import random
import statistics
import time
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    SupportsFloat,
    SupportsInt,
    SupportsIndex,
    Tuple,
    TypeAlias,
    cast,
)

JaxArray: TypeAlias = Any
NDArrayFloat: TypeAlias = Any


_np: ModuleType | None
try:  # Optional scientific backends
    _np = importlib.import_module("numpy")
except Exception:  # pragma: no cover - numpy is optional
    _np = None

_jax: ModuleType | None = None
_jnp: ModuleType | None = None
_jax_jit: Callable[..., Any] | None = None
_jax_vmap: Callable[..., Any] | None = None
_jax_device_get: Callable[[Any], Any] | None = None
_jsp_special: ModuleType | None = None
try:  # pragma: no cover - optional JAX acceleration backend
    jax_module = importlib.import_module("jax")
    jnp_module = importlib.import_module("jax.numpy")
    jsp_module = importlib.import_module("jax.scipy.special")
    _jax = jax_module
    _jnp = jnp_module
    _jsp_special = jsp_module
    _jax_device_get = getattr(jax_module, "device_get")
    _jax_jit = getattr(jax_module, "jit")
    _jax_vmap = getattr(jax_module, "vmap")
except Exception:  # pragma: no cover - optional dependency
    _jax = None
    _jnp = None
    _jax_jit = None
    _jax_vmap = None
    _jax_device_get = None
    _jsp_special = None

_numba: ModuleType | None
try:  # pragma: no cover - numba accelerates vectorised loops when available
    _numba = importlib.import_module("numba")
except Exception:  # pragma: no cover - optional dependency
    _numba = None


def _python_factorial(n: int) -> float:
    result = 1.0
    for value in range(2, n + 1):
        result *= value
    return result


def _python_bivariate_poisson_kernel(
    lam1: float,
    lam2: float,
    lam3: float,
    max_home: int,
    max_away: int,
    factorial: Callable[[int], float],
    np_module: ModuleType,
) -> NDArrayFloat:
    grid = np_module.zeros((max_home + 1, max_away + 1), dtype=np_module.float64)
    base = math.exp(-(lam1 + lam2 + lam3))
    for h in range(max_home + 1):
        for a in range(max_away + 1):
            total = 0.0
            limit = min(h, a)
            for k in range(limit + 1):
                total += (
                    base
                    * (lam1 ** (h - k))
                    / factorial(h - k)
                    * (lam2 ** (a - k))
                    / factorial(a - k)
                    * (lam3**k)
                    / factorial(k)
                )
            grid[h, a] = total
    return grid


_numba_factorial: Callable[[int], float] | None = None
_numba_bivariate_poisson_kernel: (
    Callable[[float, float, float, int, int], NDArrayFloat] | None
) = None
if _numba is not None and _np is not None:  # pragma: no cover - compiled at runtime
    njit = getattr(_numba, "njit")

    def _numba_factorial_impl(n: int) -> float:
        return _python_factorial(n)

    compiled_factorial = njit(cache=True)(_numba_factorial_impl)

    def _numba_kernel_impl(
        lam1: float, lam2: float, lam3: float, max_home: int, max_away: int
    ) -> NDArrayFloat:
        assert _np is not None
        return _python_bivariate_poisson_kernel(
            lam1, lam2, lam3, max_home, max_away, compiled_factorial, _np
        )

    _numba_factorial = compiled_factorial
    _numba_bivariate_poisson_kernel = njit(cache=True)(_numba_kernel_impl)


_jax_bivariate_poisson_kernel: (
    Callable[[float, float, float, int, int], JaxArray] | None
) = None

if (
    _jnp is not None
    and _jax_jit is not None
    and _jax_vmap is not None
    and _jsp_special is not None
):

    def _jax_bivariate_poisson_kernel_impl(
        lam1: float, lam2: float, lam3: float, max_home: int, max_away: int
    ) -> JaxArray:
        assert _jnp is not None and _jsp_special is not None and _jax_vmap is not None
        base = _jnp.exp(-(lam1 + lam2 + lam3))

        def _factorial_term(value: JaxArray) -> JaxArray:
            return _jnp.exp(_jsp_special.gammaln(value + 1.0))

        def compute_entry(home_count: int, away_count: int) -> JaxArray:
            assert _jnp is not None and _jax_vmap is not None
            limit = _jnp.minimum(home_count, away_count)
            ks = _jnp.arange(limit + 1, dtype=_jnp.int32)
            home_exponent = _jnp.maximum(0, home_count - ks).astype(_jnp.float64)
            away_exponent = _jnp.maximum(0, away_count - ks).astype(_jnp.float64)
            shared_exponent = ks.astype(_jnp.float64)
            home_term = _jnp.power(lam1, home_exponent)
            away_term = _jnp.power(lam2, away_exponent)
            shared_term = _jnp.power(lam3, shared_exponent)
            denom_home = _factorial_term(home_exponent)
            denom_away = _factorial_term(away_exponent)
            denom_shared = _factorial_term(shared_exponent)
            term = base * home_term * away_term * shared_term
            term /= denom_home * denom_away * denom_shared
            return _jnp.sum(term)

        home_range = _jnp.arange(max_home + 1)
        away_range = _jnp.arange(max_away + 1)
        row_fn = lambda h: _jax_vmap(lambda a: compute_entry(h, a))(away_range)
        grid = _jax_vmap(row_fn)(home_range)
        return grid

    _jax_bivariate_poisson_kernel = cast(
        Callable[[float, float, float, int, int], JaxArray],
        _jax_jit(static_argnums=(3, 4))(_jax_bivariate_poisson_kernel_impl),
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team and schedule primitives
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class TeamRating:
    """Represents a team's strength profile used in simulations."""

    team: str
    offensive_rating: float
    defensive_rating: float


@dataclasses.dataclass(slots=True)
class HistoricalGameRecord:
    """Historical features used to calibrate scoring rate models."""

    home_team: str
    away_team: str
    home_points: int
    away_points: int
    home_pace: float
    away_pace: float
    home_offense_rating: float
    home_defense_rating: float
    away_offense_rating: float
    away_defense_rating: float


def build_historical_records(
    history: Iterable[HistoricalGameRecord | Mapping[str, object] | object],
) -> List[HistoricalGameRecord]:
    """Normalise iterable data into :class:`HistoricalGameRecord` instances.

    The calibrator accepts flexible history inputs sourced from Polars/Pandas
    frames or dictionaries.  This helper keeps the constructor ergonomic while
    ensuring downstream code works with a consistent type.
    """

    records: List[HistoricalGameRecord] = []
    for row in history:
        if isinstance(row, HistoricalGameRecord):
            records.append(row)
            continue
        if isinstance(row, Mapping):
            values: Mapping[str, object] = row
        else:
            values = {
                "home_team": getattr(row, "home_team"),
                "away_team": getattr(row, "away_team"),
                "home_points": getattr(row, "home_points"),
                "away_points": getattr(row, "away_points"),
                "home_pace": getattr(row, "home_pace", 60.0),
                "away_pace": getattr(row, "away_pace", 60.0),
                "home_offense_rating": getattr(row, "home_offense_rating", 0.0),
                "home_defense_rating": getattr(row, "home_defense_rating", 0.0),
                "away_offense_rating": getattr(row, "away_offense_rating", 0.0),
                "away_defense_rating": getattr(row, "away_defense_rating", 0.0),
            }
        records.append(
            HistoricalGameRecord(
                home_team=_coerce_str(values.get("home_team"), "home_team"),
                away_team=_coerce_str(values.get("away_team"), "away_team"),
                home_points=_coerce_int(values.get("home_points"), "home_points"),
                away_points=_coerce_int(values.get("away_points"), "away_points"),
                home_pace=_coerce_float(values.get("home_pace"), "home_pace", 60.0),
                away_pace=_coerce_float(values.get("away_pace"), "away_pace", 60.0),
                home_offense_rating=_coerce_float(
                    values.get("home_offense_rating"), "home_offense_rating", 0.0
                ),
                home_defense_rating=_coerce_float(
                    values.get("home_defense_rating"), "home_defense_rating", 0.0
                ),
                away_offense_rating=_coerce_float(
                    values.get("away_offense_rating"), "away_offense_rating", 0.0
                ),
                away_defense_rating=_coerce_float(
                    values.get("away_defense_rating"), "away_defense_rating", 0.0
                ),
            )
        )
    return records


@dataclasses.dataclass(slots=True)
class GameSimulationConfig:
    """Configuration for the simulation engine."""

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


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------


def _distribution_moments(distribution: Mapping[int, float]) -> tuple[float, float]:
    total = float(sum(distribution.values()))
    if total <= 0.0:
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


def _safe_log(value: float) -> float:
    return math.log(max(value, 1e-6))


def _hash_token(token: str) -> float:
    return (abs(hash(token)) % 10_000) / 10_000.0


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


def _coerce_str(value: object | None, field: str) -> str:
    if value is None:
        raise TypeError(f"Missing required field {field}")
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_optional_str(value: object | None, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_int(value: object | None, field: str) -> int:
    if value is None:
        raise TypeError(f"Missing required field {field}")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    if isinstance(value, (SupportsInt, SupportsIndex)):
        return int(value)
    raise TypeError(
        f"Field {field} expected int-compatible value, got {type(value).__name__}"
    )


def _coerce_float(value: object | None, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    raise TypeError(
        f"Field {field} expected float-compatible value, got {type(value).__name__}"
    )


def _coerce_str_sequence(value: object | None, field: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise TypeError(
        f"Field {field} must be an iterable of strings, got {type(value).__name__}"
    )


def _coerce_mapping(value: object | None) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"Expected mapping, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


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
    margin_distribution: Mapping[int, float]
    total_distribution: Mapping[int, float]
    home_score_distribution: Mapping[int, float]
    away_score_distribution: Mapping[int, float]
    home_rate: float = 0.0
    away_rate: float = 0.0
    shared_rate: float = 0.0
    home_mean: float = 0.0
    away_mean: float = 0.0
    home_variance: float = 0.0
    away_variance: float = 0.0
    correlation_matrix: Mapping[tuple[str, str], float] = dataclasses.field(
        default_factory=dict
    )

    def moneyline_probability(self, team: str) -> float:
        if team == self.home_team:
            return self.home_win_probability
        if team == self.away_team:
            return self.away_win_probability
        raise KeyError(f"Team {team} not part of simulation {self.event_id}")

    def tie_probability(self) -> float:
        total = float(sum(self.margin_distribution.values()))
        if total <= 0.0:
            return 0.0
        return self.margin_distribution.get(0, 0.0) / total

    def spread_probability(
        self, team: str, line: float, scope: str = "game"
    ) -> ProbabilityTriple:
        if scope == "game":
            return self._spread_probability_game(team, line)
        return self._spread_probability_scaled(team, line, scope)

    def total_probability(
        self, side: str, line: float, scope: str = "game"
    ) -> ProbabilityTriple:
        if scope == "game":
            return self._total_probability_game(side, line)
        return self._total_probability_scaled(side, line, scope)

    def team_total_probability(
        self, team: str, side: str, line: float, scope: str = "game"
    ) -> ProbabilityTriple:
        if scope == "game":
            return self._team_total_probability_game(team, side, line)
        return self._team_total_probability_scaled(team, side, line, scope)

    def correlation(self, key_a: tuple[str, str], key_b: tuple[str, str]) -> float:
        if key_a == key_b:
            return 1.0
        pair = (key_a[0], key_b[0])
        if pair in self.correlation_matrix:
            return self.correlation_matrix[pair]
        reverse = (pair[1], pair[0])
        return self.correlation_matrix.get(reverse, 0.0)

    def _spread_probability_game(self, team: str, line: float) -> ProbabilityTriple:
        wins = 0.0
        pushes = 0.0
        total = float(sum(self.margin_distribution.values()))
        if total <= 0.0:
            return ProbabilityTriple(0.5, 0.0)
        for margin, count in self.margin_distribution.items():
            outcome = self._spread_outcome_value(team, line, margin)
            if outcome > 0:
                wins += count
            elif outcome == 0:
                pushes += count
        return ProbabilityTriple(max(0.0, min(1.0, wins / total)), pushes / total)

    def _spread_probability_scaled(
        self, team: str, line: float, scope: str
    ) -> ProbabilityTriple:
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
        wins = 0.0
        pushes = 0.0
        total_count = float(sum(self.total_distribution.values()))
        if total_count <= 0.0:
            return ProbabilityTriple(0.5, 0.0)
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
        return ProbabilityTriple(
            max(0.0, min(1.0, wins / total_count)), pushes / total_count
        )

    def _total_probability_scaled(
        self, side: str, line: float, scope: str
    ) -> ProbabilityTriple:
        mean, variance = _distribution_moments(self.total_distribution)
        factor = _scope_factor(scope)
        adj_mean = mean * factor
        adj_stdev = math.sqrt(variance * max(factor, 1e-6))
        if side == "over":
            win = 1.0 - _normal_cdf(line, adj_mean, adj_stdev)
        else:
            win = _normal_cdf(line, adj_mean, adj_stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def _team_total_probability_game(
        self, team: str, side: str, line: float
    ) -> ProbabilityTriple:
        distribution = (
            self.home_score_distribution
            if team == self.home_team
            else self.away_score_distribution
        )
        wins = 0.0
        pushes = 0.0
        total = float(sum(distribution.values()))
        if total <= 0.0:
            return ProbabilityTriple(0.5, 0.0)
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
        return ProbabilityTriple(max(0.0, min(1.0, wins / total)), pushes / total)

    def _team_total_probability_scaled(
        self, team: str, side: str, line: float, scope: str
    ) -> ProbabilityTriple:
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


# ---------------------------------------------------------------------------
# Bivariate Poisson calibration utilities
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class BivariatePoissonCalibration:
    home_intercept: float
    home_coefficients: Sequence[float]
    away_intercept: float
    away_coefficients: Sequence[float]
    mean_offense: float
    mean_defense: float
    mean_log_pace: float
    baseline_pace: float
    shared_rate: float
    team_home_pace: Mapping[str, float]
    team_away_pace: Mapping[str, float]


@dataclasses.dataclass(slots=True)
class BivariatePoissonParameters:
    lambda_home: float
    lambda_away: float
    lambda_shared: float
    pace_home: float
    pace_away: float


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    for i in range(n):
        pivot = i
        max_val = abs(a[i][i])
        for j in range(i + 1, n):
            candidate = abs(a[j][i])
            if candidate > max_val:
                pivot = j
                max_val = candidate
        if max_val <= 1e-12:
            a[i][i] = max(a[i][i], 1e-6)
            max_val = abs(a[i][i])
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
        pivot_val = a[i][i]
        if abs(pivot_val) <= 1e-12:
            pivot_val = 1e-6
        inv_pivot = 1.0 / pivot_val
        for j in range(i, n):
            a[i][j] *= inv_pivot
        b[i] *= inv_pivot
        for j in range(n):
            if j == i:
                continue
            factor = a[j][i]
            if factor == 0.0:
                continue
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]
    return b


def _linear_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    weights: Sequence[float] | None = None,
) -> List[float]:
    if not features:
        raise ValueError("Linear regression requires at least one observation")
    augmented = [[1.0, *row] for row in features]
    n_features = len(augmented[0])
    xtx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    xty = [0.0 for _ in range(n_features)]
    for idx, row in enumerate(augmented):
        weight = weights[idx] if weights else 1.0
        target = targets[idx]
        for i in range(n_features):
            xty[i] += weight * row[i] * target
            for j in range(n_features):
                xtx[i][j] += weight * row[i] * row[j]
    coefficients = _solve_linear_system(xtx, xty)
    return coefficients


class BivariatePoissonCalibrator:
    """Fits a bivariate Poisson scoring model from historical features."""

    def __init__(self, history: Sequence[HistoricalGameRecord]) -> None:
        self.history = list(history)
        if not self.history:
            raise ValueError("Calibration requires at least one historical game")
        self._calibration: BivariatePoissonCalibration | None = None

    def fit(self) -> BivariatePoissonCalibration:
        if self._calibration is not None:
            return self._calibration
        home_features: List[List[float]] = []
        home_targets: List[float] = []
        away_features: List[List[float]] = []
        away_targets: List[float] = []
        home_weights: List[float] = []
        away_weights: List[float] = []
        mean_offense = statistics.fmean(
            game.home_offense_rating for game in self.history
        )
        mean_defense = statistics.fmean(
            game.home_defense_rating for game in self.history
        )
        mean_log_pace = statistics.fmean(
            _safe_log((game.home_pace + game.away_pace) / 2.0) for game in self.history
        )
        baseline_pace = statistics.fmean(
            (game.home_pace + game.away_pace) / 2.0 for game in self.history
        )
        team_home_pace: MutableMapping[str, List[float]] = collections.defaultdict(list)
        team_away_pace: MutableMapping[str, List[float]] = collections.defaultdict(list)
        for game in self.history:
            home_features.append(
                [
                    game.home_offense_rating - mean_offense,
                    game.away_defense_rating - mean_defense,
                    _safe_log(game.home_pace) - mean_log_pace,
                    1.0,
                ]
            )
            home_targets.append(_safe_log(game.home_points))
            home_weights.append(max(1.0, game.home_pace / baseline_pace))
            away_features.append(
                [
                    game.away_offense_rating - mean_offense,
                    game.home_defense_rating - mean_defense,
                    _safe_log(game.away_pace) - mean_log_pace,
                    0.0,
                ]
            )
            away_targets.append(_safe_log(game.away_points))
            away_weights.append(max(1.0, game.away_pace / baseline_pace))
            team_home_pace[game.home_team].append(game.home_pace)
            team_away_pace[game.away_team].append(game.away_pace)
        home_coeffs = _linear_regression(home_features, home_targets, home_weights)
        away_coeffs = _linear_regression(away_features, away_targets, away_weights)
        home_intercept, home_beta = home_coeffs[0], home_coeffs[1:]
        away_intercept, away_beta = away_coeffs[0], away_coeffs[1:]
        home_mean = statistics.fmean(game.home_points for game in self.history)
        away_mean = statistics.fmean(game.away_points for game in self.history)
        cov = statistics.fmean(
            (game.home_points - home_mean) * (game.away_points - away_mean)
            for game in self.history
        )
        shared_rate = max(1e-6, cov)
        calibration = BivariatePoissonCalibration(
            home_intercept=home_intercept,
            home_coefficients=tuple(home_beta),
            away_intercept=away_intercept,
            away_coefficients=tuple(away_beta),
            mean_offense=mean_offense,
            mean_defense=mean_defense,
            mean_log_pace=mean_log_pace,
            baseline_pace=baseline_pace,
            shared_rate=shared_rate,
            team_home_pace={
                team: statistics.fmean(values)
                for team, values in team_home_pace.items()
            },
            team_away_pace={
                team: statistics.fmean(values)
                for team, values in team_away_pace.items()
            },
        )
        self._calibration = calibration
        return calibration

    def estimate_parameters(
        self, home_rating: TeamRating, away_rating: TeamRating
    ) -> BivariatePoissonParameters:
        calibration = self.fit()
        pace_home = calibration.team_home_pace.get(
            home_rating.team, calibration.baseline_pace
        )
        pace_away = calibration.team_away_pace.get(
            away_rating.team, calibration.baseline_pace
        )
        features_home = [
            home_rating.offensive_rating - calibration.mean_offense,
            away_rating.defensive_rating - calibration.mean_defense,
            _safe_log(pace_home) - calibration.mean_log_pace,
            1.0,
        ]
        features_away = [
            away_rating.offensive_rating - calibration.mean_offense,
            home_rating.defensive_rating - calibration.mean_defense,
            _safe_log(pace_away) - calibration.mean_log_pace,
            0.0,
        ]
        log_lambda_home = calibration.home_intercept
        for coeff, value in zip(calibration.home_coefficients, features_home):
            log_lambda_home += coeff * value
        log_lambda_away = calibration.away_intercept
        for coeff, value in zip(calibration.away_coefficients, features_away):
            log_lambda_away += coeff * value
        lambda_home = max(0.5, math.exp(log_lambda_home))
        lambda_away = max(0.5, math.exp(log_lambda_away))
        pace_scale = math.sqrt(pace_home * pace_away) / max(
            1.0, calibration.baseline_pace
        )
        lambda_shared = max(1e-6, calibration.shared_rate * pace_scale)
        return BivariatePoissonParameters(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            lambda_shared=lambda_shared,
            pace_home=pace_home,
            pace_away=pace_away,
        )


# ---------------------------------------------------------------------------
# Simulation benchmarks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class SimulationBenchmark:
    backend: str
    simulations_run: int
    elapsed_seconds: float

    @property
    def per_minute(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return float("inf")
        return self.simulations_run / (self.elapsed_seconds / 60.0)


# ---------------------------------------------------------------------------
# Bivariate Poisson engine
# ---------------------------------------------------------------------------


class BivariatePoissonEngine:
    """Analytical bivariate Poisson simulator with Skellam margins."""

    def __init__(
        self,
        ratings: Mapping[str, TeamRating],
        config: GameSimulationConfig | None = None,
        historical_games: Sequence[HistoricalGameRecord | Mapping[str, object]]
        | None = None,
        backend: str = "auto",
    ) -> None:
        self.ratings = dict(ratings)
        self.config = config or GameSimulationConfig()
        self._rng = random.Random(self.config.seed)
        history = (
            self._synthetic_history()
            if historical_games is None
            else build_historical_records(historical_games)
        )
        self.calibrator = BivariatePoissonCalibrator(history)
        self._numpy = _np if _np is not None else None
        self._numba_kernel = (
            _numba_bivariate_poisson_kernel
            if _numba_bivariate_poisson_kernel is not None
            else None
        )
        self._jax_kernel = (
            _jax_bivariate_poisson_kernel
            if _jax_bivariate_poisson_kernel is not None
            else None
        )
        requested = backend.lower()
        self._backend = "python"
        if requested not in {"auto", "python", "numpy", "numba", "jax"}:
            raise ValueError(f"Unsupported backend {backend!r}")
        if requested in {"auto", "numba"} and self._numba_kernel is not None:
            self._backend = "numba"
        elif requested in {"auto", "jax"} and self._jax_kernel is not None:
            self._backend = "jax"
        elif requested in {"auto", "numpy"} and self._numpy is not None:
            self._backend = "numpy"
        else:
            self._backend = "python"
            if requested == "numba":
                logger.warning(
                    "Numba backend requested but unavailable; falling back to python"
                )
            if requested == "jax" and self._jax_kernel is None:
                logger.warning(
                    "JAX backend requested but unavailable; falling back to python"
                )
            if requested == "numpy" and self._numpy is None:
                logger.warning(
                    "Requested numpy backend but numpy is unavailable; falling back to python"
                )
            if requested == "auto" and self._numpy is None:
                logger.debug("NumPy backend not available; using python loops")

    # -- calibration helpers -------------------------------------------------

    def _synthetic_history(self) -> List[HistoricalGameRecord]:
        teams = list(self.ratings.values())
        if len(teams) < 2:
            raise ValueError("At least two teams are required to synthesise history")
        baseline_pace = 62.0
        history: List[HistoricalGameRecord] = []
        for home in teams:
            for away in teams:
                if home.team == away.team:
                    continue
                pace_home = max(
                    50.0,
                    baseline_pace
                    + 1.2 * home.offensive_rating
                    - 0.4 * away.defensive_rating,
                )
                pace_away = max(
                    50.0,
                    baseline_pace
                    + 1.2 * away.offensive_rating
                    - 0.4 * home.defensive_rating,
                )
                home_points = max(
                    3,
                    int(
                        round(
                            23.0
                            + 0.8 * home.offensive_rating
                            - 0.7 * away.defensive_rating
                            + 0.05 * (pace_home - baseline_pace)
                        )
                    ),
                )
                away_points = max(
                    3,
                    int(
                        round(
                            21.0
                            + 0.8 * away.offensive_rating
                            - 0.7 * home.defensive_rating
                            + 0.05 * (pace_away - baseline_pace)
                        )
                    ),
                )
                history.append(
                    HistoricalGameRecord(
                        home_team=home.team,
                        away_team=away.team,
                        home_points=home_points,
                        away_points=away_points,
                        home_pace=pace_home,
                        away_pace=pace_away,
                        home_offense_rating=home.offensive_rating,
                        home_defense_rating=home.defensive_rating,
                        away_offense_rating=away.offensive_rating,
                        away_defense_rating=away.defensive_rating,
                    )
                )
        return history

    # -- simulation API -----------------------------------------------------

    def simulate_game(
        self, event_id: str, home_team: str, away_team: str
    ) -> SimulationResult:
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        params = self.calibrator.estimate_parameters(home_rating, away_rating)
        joint_distribution = self._joint_score_distribution(params)
        margin_distribution: MutableMapping[int, float] = collections.defaultdict(float)
        total_distribution: MutableMapping[int, float] = collections.defaultdict(float)
        home_distribution: MutableMapping[int, float] = collections.defaultdict(float)
        away_distribution: MutableMapping[int, float] = collections.defaultdict(float)
        home_win_probability = 0.0
        away_win_probability = 0.0
        expected_margin = 0.0
        expected_total = 0.0
        home_mean = 0.0
        away_mean = 0.0
        home_sq = 0.0
        away_sq = 0.0
        cross = 0.0
        for (home_score, away_score), probability in joint_distribution.items():
            margin = home_score - away_score
            total = home_score + away_score
            margin_distribution[margin] += probability
            total_distribution[total] += probability
            home_distribution[home_score] += probability
            away_distribution[away_score] += probability
            expected_margin += margin * probability
            expected_total += total * probability
            home_mean += home_score * probability
            away_mean += away_score * probability
            home_sq += (home_score**2) * probability
            away_sq += (away_score**2) * probability
            cross += home_score * away_score * probability
            if margin > 0:
                home_win_probability += probability
            elif margin < 0:
                away_win_probability += probability
            else:
                home_win_probability += 0.5 * probability
                away_win_probability += 0.5 * probability
        home_variance = max(1e-6, home_sq - home_mean**2)
        away_variance = max(1e-6, away_sq - away_mean**2)
        covariance = cross - home_mean * away_mean
        correlation = 0.0
        if home_variance > 0.0 and away_variance > 0.0:
            correlation = covariance / math.sqrt(home_variance * away_variance)
        total_stdev = math.sqrt(home_variance + away_variance)
        correlation_matrix: Dict[tuple[str, str], float] = {
            ("home_score", "away_score"): correlation,
            ("away_score", "home_score"): correlation,
            ("home_score", "total"): math.sqrt(home_variance) / max(1e-6, total_stdev),
            ("total", "home_score"): math.sqrt(home_variance) / max(1e-6, total_stdev),
            ("away_score", "total"): math.sqrt(away_variance) / max(1e-6, total_stdev),
            ("total", "away_score"): math.sqrt(away_variance) / max(1e-6, total_stdev),
        }
        iterations = self.config.iterations
        result = SimulationResult(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            iterations=iterations,
            home_win_probability=home_win_probability,
            away_win_probability=away_win_probability,
            expected_margin=expected_margin,
            expected_total=expected_total,
            margin_distribution=dict(margin_distribution),
            total_distribution=dict(total_distribution),
            home_score_distribution=dict(home_distribution),
            away_score_distribution=dict(away_distribution),
            home_rate=params.lambda_home,
            away_rate=params.lambda_away,
            shared_rate=params.lambda_shared,
            home_mean=home_mean,
            away_mean=away_mean,
            home_variance=home_variance,
            away_variance=away_variance,
            correlation_matrix=correlation_matrix,
        )
        logger.debug(
            "Bivariate Poisson simulation %s vs %s -> margin %.2f total %.2f",
            home_team,
            away_team,
            expected_margin,
            expected_total,
        )
        return result

    def simulate_many(
        self, fixtures: Iterable[tuple[str, str, str]]
    ) -> List[SimulationResult]:
        return [
            self.simulate_game(event_id, home_team, away_team)
            for event_id, home_team, away_team in fixtures
        ]

    def benchmark(
        self, games: Sequence[tuple[str, str, str]], repeats: int = 3
    ) -> SimulationBenchmark:
        start = time.perf_counter()
        simulations = 0
        for _ in range(repeats):
            for event_id, home, away in games:
                self.simulate_game(event_id, home, away)
                simulations += 1
        elapsed = time.perf_counter() - start
        backend = (
            self._backend if self._backend in {"numpy", "numba", "jax"} else "python"
        )
        benchmark = SimulationBenchmark(
            backend=backend, simulations_run=simulations, elapsed_seconds=elapsed
        )
        if benchmark.per_minute < 1_000_000:
            if self._backend == "python" and self._numpy is not None:
                suggestion = "numpy"
            elif self._backend == "python" and self._jax_kernel is not None:
                suggestion = "jax"
            else:
                suggestion = "numba"
            logger.debug(
                "Benchmark throughput %.0f sims/minute below target; consider enabling %s backend",
                benchmark.per_minute,
                suggestion,
            )
        return benchmark

    # -- backend implementations -------------------------------------------

    def _joint_score_distribution(
        self, params: BivariatePoissonParameters
    ) -> Dict[tuple[int, int], float]:
        max_home = int(
            math.ceil(
                params.lambda_home
                + params.lambda_shared
                + 6 * math.sqrt(params.lambda_home + params.lambda_shared)
            )
        )
        max_away = int(
            math.ceil(
                params.lambda_away
                + params.lambda_shared
                + 6 * math.sqrt(params.lambda_away + params.lambda_shared)
            )
        )
        max_home = max(max_home, 20)
        max_away = max(max_away, 20)
        if self._backend == "numba" and self._numba_kernel is not None:
            return self._joint_distribution_numba(params, max_home, max_away)
        if self._backend == "jax" and self._jax_kernel is not None:
            return self._joint_distribution_jax(params, max_home, max_away)
        if self._backend == "numpy" and self._numpy is not None:
            return self._joint_distribution_numpy(params, max_home, max_away)
        return self._joint_distribution_python(params, max_home, max_away)

    def _joint_distribution_python(
        self,
        params: BivariatePoissonParameters,
        max_home: int,
        max_away: int,
    ) -> Dict[tuple[int, int], float]:
        lam1 = params.lambda_home
        lam2 = params.lambda_away
        lam3 = params.lambda_shared
        base = math.exp(-(lam1 + lam2 + lam3))
        factorial_cache: Dict[int, float] = {0: 1.0}

        def fact(n: int) -> float:
            if n not in factorial_cache:
                factorial_cache[n] = math.factorial(n)
            return factorial_cache[n]

        distribution: Dict[tuple[int, int], float] = {}
        for home in range(max_home + 1):
            for away in range(max_away + 1):
                total = 0.0
                limit = min(home, away)
                for k in range(limit + 1):
                    term = (
                        base
                        * (lam1 ** (home - k))
                        / fact(home - k)
                        * (lam2 ** (away - k))
                        / fact(away - k)
                        * (lam3**k)
                        / fact(k)
                    )
                    total += term
                distribution[(home, away)] = total
        return distribution

    def _joint_distribution_jax(
        self,
        params: BivariatePoissonParameters,
        max_home: int,
        max_away: int,
    ) -> Dict[tuple[int, int], float]:
        kernel = self._jax_kernel
        if kernel is None:
            return self._joint_distribution_python(params, max_home, max_away)
        grid = kernel(
            float(params.lambda_home),
            float(params.lambda_away),
            float(params.lambda_shared),
            int(max_home),
            int(max_away),
        )
        if _jax_device_get is not None:
            grid = _jax_device_get(grid)
        distribution: Dict[tuple[int, int], float] = {}
        total = 0.0
        for home in range(grid.shape[0]):
            for away in range(grid.shape[1]):
                probability = float(grid[home, away])
                if probability <= 0.0:
                    continue
                distribution[(int(home), int(away))] = probability
                total += probability
        if total > 0.0:
            scale = 1.0 / total
            for key in list(distribution.keys()):
                distribution[key] *= scale
        return distribution

    def _joint_distribution_numba(
        self,
        params: BivariatePoissonParameters,
        max_home: int,
        max_away: int,
    ) -> Dict[tuple[int, int], float]:
        kernel = self._numba_kernel
        if kernel is None:
            return self._joint_distribution_python(params, max_home, max_away)
        grid = kernel(
            float(params.lambda_home),
            float(params.lambda_away),
            float(params.lambda_shared),
            int(max_home),
            int(max_away),
        )
        distribution: Dict[tuple[int, int], float] = {}
        total = 0.0
        for home in range(grid.shape[0]):
            for away in range(grid.shape[1]):
                probability = float(grid[home, away])
                if probability <= 0.0:
                    continue
                distribution[(int(home), int(away))] = probability
                total += probability
        if total > 0.0:
            scale = 1.0 / total
            for key in list(distribution.keys()):
                distribution[key] *= scale
        return distribution

    def _joint_distribution_numpy(
        self,
        params: BivariatePoissonParameters,
        max_home: int,
        max_away: int,
    ) -> Dict[tuple[int, int], float]:
        np = self._numpy
        assert np is not None
        lam1 = params.lambda_home
        lam2 = params.lambda_away
        lam3 = params.lambda_shared
        home = np.arange(0, max_home + 1)
        away = np.arange(0, max_away + 1)
        factorials = np.vectorize(math.factorial)
        base = math.exp(-(lam1 + lam2 + lam3))
        distribution: Dict[tuple[int, int], float] = {}
        for h in home:
            for a in away:
                limit = int(min(h, a))
                ks = np.arange(0, limit + 1)
                term = (
                    base
                    * (lam1 ** (h - ks))
                    / factorials(h - ks)
                    * (lam2 ** (a - ks))
                    / factorials(a - ks)
                    * (lam3**ks)
                    / factorials(ks)
                )
                distribution[(int(h), int(a))] = float(np.sum(term))
        return distribution


# ---------------------------------------------------------------------------
# Player level models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class PlayerProjection:
    player: str
    market: str
    mean: float
    stdev: float
    distribution: str = "normal"


@dataclasses.dataclass(slots=True)
class PlayerFeatureRow:
    player: str
    opponent: str
    market: str
    scope: str
    target: float
    injury_status: float = 0.0
    weather: float = 0.0
    pace: float = 0.0
    weight: float = 1.0
    game_id: str | None = None


class PlayerOutcomeModel:
    """Base class for player level predictive models."""

    def __init__(
        self, markets: Sequence[str] | None = None, distribution: str = "normal"
    ) -> None:
        self.markets = tuple(markets or [])
        self.distribution = distribution
        self._fitted = False

    def supports_market(self, market: str) -> bool:
        return not self.markets or market in self.markets

    def fit(
        self, rows: Sequence[PlayerFeatureRow]
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(
        self,
        player: str,
        market: str,
        scope: str,
        features: Mapping[str, object],
    ) -> PlayerProjection | None:  # pragma: no cover - interface
        raise NotImplementedError

    def covariance(
        self,
        player_a: str,
        player_b: str,
        market: str,
        scope: str,
        features_a: Mapping[str, object],
        features_b: Mapping[str, object],
    ) -> float:
        del player_a, player_b, market, scope, features_a, features_b
        return 0.0


class GLMPlayerModel(PlayerOutcomeModel):
    """Lightweight Gaussian GLM over hashed opponent/injury/weather features."""

    def __init__(
        self, markets: Sequence[str] | None = None, distribution: str = "normal"
    ) -> None:
        super().__init__(markets, distribution)
        self._coefficients: List[float] | None = None
        self._residual_variance: float = 25.0
        self._mean_pace: float = 0.0

    def fit(self, rows: Sequence[PlayerFeatureRow]) -> None:
        if not rows:
            raise ValueError("GLM training requires at least one row")
        features: List[List[float]] = []
        targets: List[float] = []
        weights: List[float] = []
        self._mean_pace = statistics.fmean(row.pace for row in rows) if rows else 60.0
        for row in rows:
            features.append(
                self._encode(
                    row.player, row.opponent, row.injury_status, row.weather, row.pace
                )
            )
            targets.append(row.target)
            weights.append(row.weight)
        coefficients = _linear_regression(features, targets, weights)
        self._coefficients = coefficients
        residuals: List[float] = []
        for row, encoded in zip(rows, features):
            prediction = self._predict_raw(encoded)
            residuals.append(row.target - prediction)
        if residuals:
            self._residual_variance = max(
                1e-6, statistics.fmean(res**2 for res in residuals)
            )
        self._fitted = True

    def predict(
        self,
        player: str,
        market: str,
        scope: str,
        features: Mapping[str, object],
    ) -> PlayerProjection | None:
        if not self.supports_market(market):
            return None
        encoded = self._encode(
            player,
            _coerce_optional_str(features.get("opponent")),
            _coerce_float(features.get("injury_status"), "injury_status", 0.0),
            _coerce_float(features.get("weather"), "weather", 0.0),
            _coerce_float(features.get("pace"), "pace", self._mean_pace),
        )
        raw = self._predict_raw(encoded)
        factor = _scope_factor(scope)
        mean = raw * factor
        if self.distribution == "poisson":
            lam = max(1e-6, math.exp(raw) * factor)
            mean = lam
            stdev = math.sqrt(lam)
            distribution = "poisson"
        else:
            stdev = math.sqrt(self._residual_variance) * math.sqrt(max(factor, 1e-6))
            distribution = self.distribution
        return PlayerProjection(
            player=player,
            market=market,
            mean=mean,
            stdev=stdev,
            distribution=distribution,
        )

    def covariance(
        self,
        player_a: str,
        player_b: str,
        market: str,
        scope: str,
        features_a: Mapping[str, object],
        features_b: Mapping[str, object],
    ) -> float:
        del market
        if not self._coefficients:
            return 0.0
        encoded_a = self._encode(
            player_a,
            _coerce_optional_str(features_a.get("opponent")),
            _coerce_float(features_a.get("injury_status"), "injury_status", 0.0),
            _coerce_float(features_a.get("weather"), "weather", 0.0),
            _coerce_float(features_a.get("pace"), "pace", self._mean_pace),
        )
        encoded_b = self._encode(
            player_b,
            _coerce_optional_str(features_b.get("opponent")),
            _coerce_float(features_b.get("injury_status"), "injury_status", 0.0),
            _coerce_float(features_b.get("weather"), "weather", 0.0),
            _coerce_float(features_b.get("pace"), "pace", self._mean_pace),
        )
        dot = sum(a * b for a, b in zip(encoded_a, encoded_b))
        norm_a = math.sqrt(sum(a**2 for a in encoded_a))
        norm_b = math.sqrt(sum(b**2 for b in encoded_b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        similarity = dot / (norm_a * norm_b)
        factor = _scope_factor(scope)
        return similarity * self._residual_variance * max(factor, 1e-6)

    def _encode(
        self,
        player: str,
        opponent: str,
        injury_status: float,
        weather: float,
        pace: float,
    ) -> List[float]:
        return [
            _hash_token(player),
            _hash_token(opponent),
            injury_status,
            weather,
            pace - self._mean_pace,
        ]

    def _predict_raw(self, encoded: Sequence[float]) -> float:
        if not self._coefficients:
            return 0.0
        augmented = [1.0, *encoded]
        return sum(coeff * value for coeff, value in zip(self._coefficients, augmented))


class XGBoostPlayerModel(GLMPlayerModel):
    """Gradient boosted analogue built from GLM residuals."""

    def __init__(
        self, markets: Sequence[str] | None = None, distribution: str = "normal"
    ) -> None:
        super().__init__(markets, distribution)
        self._boost_term: float = 0.0

    def fit(self, rows: Sequence[PlayerFeatureRow]) -> None:
        super().fit(rows)
        residuals = []
        for row in rows:
            base = self.predict(
                row.player,
                row.market,
                row.scope,
                {
                    "opponent": row.opponent,
                    "injury_status": row.injury_status,
                    "weather": row.weather,
                    "pace": row.pace,
                },
            )
            if base is None:
                continue
            residuals.append(row.target - base.mean)
        if residuals:
            self._boost_term = statistics.fmean(residuals)
        self._fitted = True

    def predict(
        self,
        player: str,
        market: str,
        scope: str,
        features: Mapping[str, object],
    ) -> PlayerProjection | None:
        projection = super().predict(player, market, scope, features)
        if projection is None:
            return None
        projection = dataclasses.replace(projection)
        projection.mean += self._boost_term * _scope_factor(scope)
        projection.stdev = max(1e-3, projection.stdev * 0.95)
        return projection

    def covariance(
        self,
        player_a: str,
        player_b: str,
        market: str,
        scope: str,
        features_a: Mapping[str, object],
        features_b: Mapping[str, object],
    ) -> float:
        base = super().covariance(
            player_a, player_b, market, scope, features_a, features_b
        )
        return base * 1.1


class NGBoostPlayerModel(GLMPlayerModel):
    """Normalising-flow inspired GLM with heteroscedastic variance estimates."""

    def __init__(self, markets: Sequence[str] | None = None) -> None:
        super().__init__(markets, distribution="normal")
        self._variance_coefficients: List[float] | None = None

    def fit(self, rows: Sequence[PlayerFeatureRow]) -> None:
        super().fit(rows)
        if not rows:
            return
        features = [
            self._encode(
                row.player, row.opponent, row.injury_status, row.weather, row.pace
            )
            for row in rows
        ]
        residuals = []
        for row, encoded in zip(rows, features):
            prediction = self._predict_raw(encoded)
            residuals.append((row.target - prediction) ** 2)
        self._variance_coefficients = _linear_regression(
            features, residuals, [row.weight for row in rows]
        )

    def predict(
        self,
        player: str,
        market: str,
        scope: str,
        features: Mapping[str, object],
    ) -> PlayerProjection | None:
        projection = super().predict(player, market, scope, features)
        if projection is None:
            return None
        if self._variance_coefficients:
            encoded = self._encode(
                player,
                _coerce_optional_str(features.get("opponent")),
                _coerce_float(features.get("injury_status"), "injury_status", 0.0),
                _coerce_float(features.get("weather"), "weather", 0.0),
                _coerce_float(features.get("pace"), "pace", self._mean_pace),
            )
            variance = sum(
                coeff * value
                for coeff, value in zip(self._variance_coefficients, [1.0, *encoded])
            )
            variance = max(1e-6, variance)
            projection = dataclasses.replace(
                projection,
                stdev=math.sqrt(variance * max(_scope_factor(scope), 1e-6)),
            )
        return projection

    def covariance(
        self,
        player_a: str,
        player_b: str,
        market: str,
        scope: str,
        features_a: Mapping[str, object],
        features_b: Mapping[str, object],
    ) -> float:
        base = super().covariance(
            player_a, player_b, market, scope, features_a, features_b
        )
        return base * 1.25


# ---------------------------------------------------------------------------
# Player prop forecaster orchestrating projections and covariance
# ---------------------------------------------------------------------------


class PlayerPropForecaster:
    """Advanced probabilistic model for player propositions."""

    def __init__(
        self,
        projections: Iterable[PlayerProjection] | None = None,
        models: Iterable[PlayerOutcomeModel] | None = None,
    ) -> None:
        self._projections: Dict[tuple[str, str], PlayerProjection] = {}
        self._models: Dict[str, List[PlayerOutcomeModel]] = collections.defaultdict(
            list
        )
        self._covariances: Dict[
            tuple[str, str, str], Dict[tuple[str, str, str], float]
        ] = collections.defaultdict(dict)
        if projections:
            for projection in projections:
                self.register_projection(projection)
        if models:
            for model in models:
                self.register_model(model)

    def register_projection(self, projection: PlayerProjection) -> None:
        self._projections[(projection.player, projection.market)] = projection

    def register_model(self, model: PlayerOutcomeModel) -> None:
        if not isinstance(model, PlayerOutcomeModel):
            raise TypeError("Model must inherit PlayerOutcomeModel")
        for market in model.markets or ["*"]:
            key = market if market != "*" else "*"
            self._models[key].append(model)

    def fit_pipelines(
        self,
        rows: Sequence[PlayerFeatureRow],
        markets: Sequence[str] | None = None,
        distribution: str = "normal",
    ) -> None:
        if not rows:
            raise ValueError("Player pipelines require non-empty training data")
        markets_set = set(markets or {row.market for row in rows})
        for market in markets_set:
            market_rows = [row for row in rows if row.market == market]
            if not market_rows:
                continue
            glm = GLMPlayerModel([market], distribution=distribution)
            glm.fit(market_rows)
            self.register_model(glm)
            boosted = XGBoostPlayerModel([market], distribution=distribution)
            boosted.fit(market_rows)
            self.register_model(boosted)
            ngboost = NGBoostPlayerModel([market])
            ngboost.fit(market_rows)
            self.register_model(ngboost)
        self._learn_covariances(rows)

    def register_covariance(
        self,
        player_a: str,
        market_a: str,
        scope_a: str,
        player_b: str,
        market_b: str,
        scope_b: str,
        value: float,
    ) -> None:
        key_a = (player_a, market_a, scope_a)
        key_b = (player_b, market_b, scope_b)
        self._covariances[key_a][key_b] = value
        self._covariances[key_b][key_a] = value

    def _learn_covariances(self, rows: Sequence[PlayerFeatureRow]) -> None:
        games: Dict[tuple[str, str, str], Dict[str, Tuple[float, float]]] = (
            collections.defaultdict(dict)
        )
        for row in rows:
            if not row.game_id:
                continue
            key = (row.market, row.scope, row.game_id)
            games[key][row.player] = (row.target, row.weight)
        aggregate: Dict[tuple[str, str, str, str], List[Tuple[float, float, float]]] = (
            collections.defaultdict(list)
        )
        for (market, scope, _game_id), participants in games.items():
            if len(participants) < 2:
                continue
            for (player_a, (value_a, weight_a)), (
                player_b,
                (value_b, weight_b),
            ) in itertools.combinations(participants.items(), 2):
                weight = max(1e-6, (weight_a + weight_b) / 2.0)
                aggregate[(market, scope, player_a, player_b)].append(
                    (value_a, value_b, weight)
                )
        for (market, scope, player_a, player_b), samples in aggregate.items():
            if len(samples) < 2:
                continue
            cov = self._weighted_covariance(samples)
            if cov == 0.0:
                continue
            self.register_covariance(
                player_a,
                market,
                scope,
                player_b,
                market,
                scope,
                cov,
            )

    @staticmethod
    def _weighted_covariance(samples: Sequence[Tuple[float, float, float]]) -> float:
        total_weight = sum(weight for _, _, weight in samples)
        if total_weight <= 0.0:
            return 0.0
        mean_a = sum(value_a * weight for value_a, _, weight in samples) / total_weight
        mean_b = sum(value_b * weight for _, value_b, weight in samples) / total_weight
        covariance = (
            sum(
                (value_a - mean_a) * (value_b - mean_b) * weight
                for value_a, value_b, weight in samples
            )
            / total_weight
        )
        return covariance

    def probability(
        self,
        player: str,
        market: str,
        side: str,
        line: float | None,
        scope: str,
        extra: Mapping[str, object] | None = None,
    ) -> ProbabilityTriple:
        extra_mapping: Mapping[str, object] = extra or {}
        component_names = _coerce_str_sequence(
            extra_mapping.get("components"), "components"
        )
        if component_names:
            return self._composite_probability(
                component_names,
                market,
                side,
                line,
                scope,
                extra_mapping,
            )
        projection = self._resolve_projection(player, market, scope, extra_mapping)
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
        stdev = max(1e-3, projection.stdev * math.sqrt(max(factor, 1e-6)))
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
        extra_mapping: Mapping[str, object] = extra or {}
        projection = self._resolve_projection(player, market, scope, extra_mapping)
        if projection is None:
            return 0.0, 1.0
        factor = _scope_factor(scope)
        if projection.distribution == "bernoulli":
            p = min(1.0, max(0.0, projection.mean * factor))
            stdev = math.sqrt(max(1e-6, p * (1.0 - p)))
            return p, stdev
        if projection.distribution == "poisson":
            lam = max(1e-6, projection.mean * factor)
            return lam, math.sqrt(lam)
        mean = projection.mean * factor
        stdev = max(1e-3, projection.stdev * math.sqrt(max(factor, 1e-6)))
        return mean, stdev

    def _composite_probability(
        self,
        components: Sequence[str],
        market: str,
        side: str,
        line: float | None,
        scope: str,
        extra: Mapping[str, object],
    ) -> ProbabilityTriple:
        if line is None:
            return ProbabilityTriple(0.5)
        total_mean = 0.0
        total_variance = 0.0
        stats: Dict[str, Tuple[float, float]] = {}
        raw_component_features = _coerce_mapping(extra.get("component_features"))
        component_features: Dict[str, Mapping[str, object]] = {}
        for name, value in raw_component_features.items():
            if isinstance(value, Mapping):
                component_features[str(name)] = value
        for participant in components:
            participant_extra = component_features.get(participant, extra)
            projection = self._resolve_projection(
                participant, market, scope, participant_extra
            )
            mean, stdev = self.projection_stats(
                participant, market, scope, participant_extra
            )
            stats[participant] = (mean, stdev)
            total_mean += mean
            total_variance += stdev**2
        composite_mode = _coerce_optional_str(
            extra.get("composite_mode"), "sum"
        ).lower()
        if composite_mode == "either":
            return self._composite_probability_either(
                components,
                market,
                side,
                line,
                scope,
                extra,
                stats,
            )
        for a, b in itertools.combinations(components, 2):
            cov = self._component_covariance(
                a,
                b,
                market,
                scope,
                component_features.get(a, extra),
                component_features.get(b, extra),
            )
            total_variance += 2.0 * cov
        stdev = math.sqrt(max(1e-6, total_variance))
        if side == "over":
            win = 1.0 - _normal_cdf(line, total_mean, stdev)
        else:
            win = _normal_cdf(line, total_mean, stdev)
        return ProbabilityTriple(max(0.0, min(1.0, win)))

    def _composite_probability_either(
        self,
        components: Sequence[str],
        market: str,
        side: str,
        line: float | None,
        scope: str,
        extra: Mapping[str, object],
        stats: Mapping[str, Tuple[float, float]],
    ) -> ProbabilityTriple:
        if line is None:
            return ProbabilityTriple(0.5)
        raw_component_features = _coerce_mapping(extra.get("component_features"))
        component_features: Dict[str, Mapping[str, object]] = {}
        for name, value in raw_component_features.items():
            if isinstance(value, Mapping):
                component_features[str(name)] = value
        singles: Dict[str, ProbabilityTriple] = {}
        for participant in components:
            participant_features = dict(component_features.get(participant, extra))
            participant_features.pop("components", None)
            participant_features.pop("composite_mode", None)
            participant_features.pop("participants", None)
            singles[participant] = self.probability(
                participant,
                market,
                "over" if side in {"over", "under"} else side,
                line,
                scope,
                participant_features,
            )
        if _np is None or len(components) == 1:
            base = 1.0
            for participant in components:
                base *= 1.0 - singles[participant].win
            success_probability = 1.0 - base
        else:
            assert _np is not None
            np = _np
            means = np.array([stats[name][0] for name in components], dtype=float)
            stdevs = np.array([stats[name][1] for name in components], dtype=float)
            covariance = np.eye(len(components), dtype=float)
            for idx, name in enumerate(components):
                covariance[idx, idx] = max(1e-6, stdevs[idx] ** 2)
            for i, name_a in enumerate(components):
                for j, name_b in enumerate(components):
                    if j <= i:
                        continue
                    cov = self._component_covariance(
                        name_a,
                        name_b,
                        market,
                        scope,
                        component_features.get(name_a, extra),
                        component_features.get(name_b, extra),
                    )
                    covariance[i, j] = covariance[j, i] = cov
            jitter = np.eye(len(components), dtype=float) * 1e-6
            covariance = covariance + jitter
            seed = abs(hash((tuple(components), market, scope))) % (2**32)
            rng = np.random.default_rng(seed)
            if "simulation_samples" in extra:
                samples = _coerce_int(
                    extra.get("simulation_samples"), "simulation_samples"
                )
            else:
                samples = 4096
            draws = rng.multivariate_normal(means, covariance, size=max(1, samples))
            successes = (draws >= line).any(axis=1)
            success_probability = float(np.mean(successes))
        win = (
            success_probability
            if side not in {"under", "no"}
            else 1.0 - success_probability
        )
        win = max(0.0, min(1.0, win))
        return ProbabilityTriple(win)

    def _component_covariance(
        self,
        player_a: str,
        player_b: str,
        market: str,
        scope: str,
        extra_a: Mapping[str, object],
        extra_b: Mapping[str, object],
    ) -> float:
        key_a = (player_a, market, scope)
        key_b = (player_b, market, scope)
        direct = self._covariances.get(key_a, {}).get(key_b)
        if direct is not None:
            return direct
        models = self._models.get(market, []) + self._models.get("*", [])
        covariances = [
            model.covariance(player_a, player_b, market, scope, extra_a, extra_b)
            for model in models
        ]
        covariances = [cov for cov in covariances if cov]
        if covariances:
            return statistics.fmean(covariances)
        return 0.0

    def _resolve_projection(
        self,
        player: str,
        market: str,
        scope: str,
        extra: Mapping[str, object] | None,
    ) -> PlayerProjection | None:
        key = (player, market)
        if key in self._projections:
            return self._projections[key]
        base_market = market[:-4] if market.endswith("_alt") else market
        projection = self._projections.get((player, base_market))
        if projection:
            return projection
        features: Mapping[str, object] = extra or {}
        models = self._models.get(base_market, []) + self._models.get("*", [])
        for model in models:
            if not model.supports_market(base_market):
                continue
            projection = model.predict(player, base_market, scope, features)
            if projection:
                self.register_projection(projection)
                return projection
        if extra and "projection_mean" in extra:
            mean = _coerce_float(extra.get("projection_mean"), "projection_mean", 0.0)
            stdev = _coerce_float(
                extra.get("projection_stdev"),
                "projection_stdev",
                max(5.0, mean * 0.2),
            )
            distribution = _coerce_optional_str(
                extra.get("projection_distribution"), "normal"
            )
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


# ---------------------------------------------------------------------------
# Compatibility layer ------------------------------------------------------
# ---------------------------------------------------------------------------


# Historically the MonteCarloEngine name was exported; keep an alias so that
# downstream imports continue to work transparently with the new deterministic
# bivariate Poisson implementation.
MonteCarloEngine = BivariatePoissonEngine


# ``random.Random`` lacked a Poisson method prior to Python 3.12; provide a
# lightweight fallback for callers that still rely on it elsewhere in the stack.
if not hasattr(random.Random, "poisson"):

    def _poisson(random_self: random.Random, lam: float) -> int:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random_self.random()
        return k - 1

    setattr(random.Random, "poisson", _poisson)
