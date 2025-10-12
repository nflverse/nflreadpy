# mypy: follow_imports=skip

"""Edge detection and bankroll sizing utilities."""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import logging
import math
import random
import statistics
import time
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
)

if TYPE_CHECKING:
    from typing import Literal

    class ProbabilityTriple:
        win: float
        push: float

        def __init__(self, win: float, push: float = ...) -> None: ...

        @property
        def loss(self) -> float: ...

    class SimulationResult(Protocol):
        event_id: str
        home_team: str
        away_team: str

        def moneyline_probability(self, team: str) -> float: ...

        def spread_probability(
            self, team: str, line: float, scope: str = ...
        ) -> ProbabilityTriple: ...

        def team_total_probability(
            self, team: str, side: str, line: float, scope: str = ...
        ) -> ProbabilityTriple: ...

        def total_probability(
            self, side: str, line: float, scope: str = ...
        ) -> ProbabilityTriple: ...

        def tie_probability(self) -> float: ...

        def correlation(self, key_a: Tuple[str, str], key_b: Tuple[str, str]) -> float: ...

    class PlayerPropForecaster:
        def probability(
            self,
            player: str,
            market: str,
            side: str,
            line: float | None,
            scope: str,
            extra: Mapping[str, object] | None = None,
        ) -> ProbabilityTriple: ...

        def projection_stats(
            self,
            player: str,
            market: str,
            scope: str,
            extra: Mapping[str, object] | None = None,
        ) -> Tuple[float, float]: ...

    class SimulationBenchmark:
        backend: str
        simulations_run: int
        elapsed_seconds: float

        def __init__(
            self, backend: str, simulations_run: int, elapsed_seconds: float
        ) -> None: ...

    class AlertManager(Protocol):
        def notify_edges(self, opportunities: Sequence[Mapping[str, object]]) -> None: ...

        def notify_line_movement(
            self,
            movements: Sequence[Mapping[str, object]],
            *,
            threshold: int = ...,
        ) -> None: ...

    def get_alert_manager() -> AlertManager: ...

    class ResponsibleGamingControls:
        session_loss_limit: float | None
        session_stake_limit: float | None
        cooldown_seconds: float | None

        def __init__(
            self,
            session_loss_limit: float | None = ...,
            session_stake_limit: float | None = ...,
            cooldown_seconds: float | None = ...,
        ) -> None: ...

    class ComplianceEngine(Protocol):
        def validate(self, opportunity: "Opportunity") -> bool: ...

    ScopeLiteral = Literal["game", "1h", "2h", "1q", "2q", "3q", "4q"]
    EntityLiteral = Literal["team", "player", "total", "either", "leader"]

    class OddsQuote:
        event_id: str
        sportsbook: str
        book_market_group: str
        market: str
        scope: ScopeLiteral
        entity_type: EntityLiteral
        team_or_player: str
        side: str | None
        line: float | None
        american_odds: int
        observed_at: dt.datetime
        extra: Mapping[str, object] | None

        def implied_probability(self) -> float: ...

        def decimal_multiplier(self) -> float: ...

        def __init__(
            self,
            *,
            event_id: str,
            sportsbook: str,
            book_market_group: str,
            market: str,
            scope: ScopeLiteral,
            entity_type: EntityLiteral,
            team_or_player: str,
            side: str | None,
            line: float | None,
            american_odds: int,
            observed_at: dt.datetime,
            extra: Mapping[str, object] | None = None,
        ) -> None: ...

    def american_to_decimal(value: int | float | str) -> float: ...

    def american_to_fractional(
        value: int, *, max_denominator: int = ...
    ) -> tuple[int, int]: ...

    def american_to_profit_multiplier(value: int | float | str) -> float: ...

    def fractional_to_decimal(numerator: int, denominator: int) -> float: ...

    def implied_probability_from_american(value: int | float | str) -> float: ...

    def implied_probability_to_decimal(probability: float) -> float: ...

    def best_prices_by_selection(
        quotes: Sequence[OddsQuote],
    ) -> Dict[Tuple[str, str, ScopeLiteral, str, str | None, float | None], OddsQuote]: ...
else:  # pragma: no cover - imported for runtime behaviour
    import importlib

    _alerts = importlib.import_module(".alerts", __package__)
    AlertManager = _alerts.AlertManager
    get_alert_manager = _alerts.get_alert_manager

    _compliance = importlib.import_module(".compliance", __package__)
    ComplianceEngine = _compliance.ComplianceEngine
    ResponsibleGamingControls = _compliance.ResponsibleGamingControls

    _models = importlib.import_module(".models", __package__)
    PlayerPropForecaster = _models.PlayerPropForecaster
    ProbabilityTriple = _models.ProbabilityTriple
    SimulationBenchmark = _models.SimulationBenchmark
    SimulationResult = _models.SimulationResult

    _scrapers = importlib.import_module(".scrapers.base", __package__)
    EntityLiteral = _scrapers.EntityLiteral
    OddsQuote = _scrapers.OddsQuote
    ScopeLiteral = _scrapers.ScopeLiteral
    american_to_decimal = _scrapers.american_to_decimal
    american_to_fractional = _scrapers.american_to_fractional
    american_to_profit_multiplier = _scrapers.american_to_profit_multiplier
    best_prices_by_selection = _scrapers.best_prices_by_selection
    fractional_to_decimal = _scrapers.fractional_to_decimal
    implied_probability_from_american = _scrapers.implied_probability_from_american
    implied_probability_to_decimal = _scrapers.implied_probability_to_decimal

if TYPE_CHECKING:
    from .ingestion import IngestedOdds as IngestedOddsType
else:  # pragma: no cover - optional import for runtime behaviour
    try:
        from .ingestion import IngestedOdds as IngestedOddsType
    except Exception:  # pragma: no cover
        IngestedOddsType = object

IngestedOdds: TypeAlias = IngestedOddsType

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class Opportunity:
    event_id: str
    sportsbook: str
    book_market_group: str
    market: str
    scope: ScopeLiteral
    entity_type: EntityLiteral
    team_or_player: str
    side: str | None
    line: float | None
    american_odds: int
    model_probability: float
    push_probability: float
    implied_probability: float
    expected_value: float
    kelly_fraction: float
    extra: Mapping[str, object]

    def decimal_odds(self) -> float:
        """Return decimal odds for the quoted American price."""

        return american_to_decimal(self.american_odds)

    def fractional_odds(self, *, max_denominator: int = 512) -> tuple[int, int]:
        """Return fractional odds matching the quoted American price."""

        return american_to_fractional(
            self.american_odds, max_denominator=max_denominator
        )

    def decimal_multiplier(self) -> float:
        """Expose the profit multiplier used for EV calculations."""

        return american_to_profit_multiplier(self.american_odds)


class KellyCriterion:
    """Utility for computing fractional Kelly bet sizes."""

    @staticmethod
    def fraction(
        win_probability: float,
        loss_probability: float,
        price: int | float | str,
        *,
        fractional_kelly: float = 1.0,
        cap: float | None = None,
    ) -> float:
        multiplier = american_to_profit_multiplier(price)
        return KellyCriterion._fraction_from_multiplier(
            win_probability,
            loss_probability,
            multiplier,
            fractional_kelly=fractional_kelly,
            cap=cap,
        )

    @staticmethod
    def fraction_from_decimal(
        win_probability: float,
        loss_probability: float,
        decimal_odds: float,
        *,
        fractional_kelly: float = 1.0,
        cap: float | None = None,
    ) -> float:
        if decimal_odds <= 1.0:
            raise ValueError("Decimal odds must exceed 1.0")
        multiplier = decimal_odds - 1.0
        return KellyCriterion._fraction_from_multiplier(
            win_probability,
            loss_probability,
            multiplier,
            fractional_kelly=fractional_kelly,
            cap=cap,
        )

    @staticmethod
    def fraction_from_fractional(
        win_probability: float,
        loss_probability: float,
        numerator: int,
        denominator: int,
        *,
        fractional_kelly: float = 1.0,
        cap: float | None = None,
    ) -> float:
        decimal = fractional_to_decimal(numerator, denominator)
        return KellyCriterion.fraction_from_decimal(
            win_probability,
            loss_probability,
            decimal,
            fractional_kelly=fractional_kelly,
            cap=cap,
        )

    @staticmethod
    def fraction_from_implied_probability(
        win_probability: float,
        loss_probability: float,
        implied_probability: float,
        *,
        fractional_kelly: float = 1.0,
        cap: float | None = None,
    ) -> float:
        decimal = implied_probability_to_decimal(implied_probability)
        return KellyCriterion.fraction_from_decimal(
            win_probability,
            loss_probability,
            decimal,
            fractional_kelly=fractional_kelly,
            cap=cap,
        )

    @staticmethod
    def _fraction_from_multiplier(
        win_probability: float,
        loss_probability: float,
        profit_multiplier: float,
        *,
        fractional_kelly: float = 1.0,
        cap: float | None = None,
    ) -> float:
        if profit_multiplier <= 0.0:
            raise ValueError("Profit multiplier must be positive")
        win = max(0.0, min(1.0, win_probability))
        loss = max(0.0, loss_probability)
        numerator = profit_multiplier * win - loss
        if numerator <= 0:
            return 0.0
        kelly = numerator / profit_multiplier
        scaled = max(0.0, kelly) * max(0.0, fractional_kelly)
        if cap is not None:
            scaled = min(cap, scaled)
        return scaled


class EdgeDetector:
    """Compare simulation results to bookmaker odds and flag edges."""

    def __init__(
        self,
        value_threshold: float = 0.02,
        player_model: PlayerPropForecaster | None = None,
        alert_manager: AlertManager | None = None,
        backend: str = "auto",
        correlation_penalty: float = 0.0,
    ) -> None:
        self.value_threshold = value_threshold
        self.player_model = player_model or PlayerPropForecaster()
        self.alert_manager = alert_manager or get_alert_manager()
        self.correlation_penalty = correlation_penalty
        self._backend = "python"
        self._np = None
        if backend in {"auto", "numpy"}:
            try:  # pragma: no cover - optional dependency
                import numpy as np  # type: ignore

                self._np = np
                self._backend = "numpy"
            except Exception:  # pragma: no cover - optional dependency
                self._np = None
                self._backend = "python"
        elif backend == "numpy":
            try:
                import numpy as np

                self._np = np
                self._backend = "numpy"
            except Exception:  # pragma: no cover
                logger.warning("NumPy backend requested but unavailable; falling back to python")
                self._backend = "python"
        else:
            self._backend = "python"

    def detect(
        self, odds: Sequence[OddsQuote], simulations: Sequence[SimulationResult]
    ) -> List[Opportunity]:
        by_event = {result.event_id: result for result in simulations}
        opportunities: List[Opportunity] = []
        grouped: Dict[str, List[OddsQuote]] = collections.defaultdict(list)
        for quote in odds:
            grouped[quote.event_id].append(quote)
        for event_id, event_quotes in grouped.items():
            result = by_event.get(event_id)
            evaluations: List[Tuple[OddsQuote, ProbabilityTriple, SimulationResult | None]] = []
            for quote in event_quotes:
                probability = self._probability_for_quote(quote, result)
                if probability is None:
                    continue
                evaluations.append((quote, probability, result))
            if not evaluations:
                continue
            adjusted_probabilities, expected_values = self._evaluate_probabilities(evaluations)
            for (quote, _base_probability, _sim_result), adj_probability, expected_value in zip(
                evaluations, adjusted_probabilities, expected_values
            ):
                if expected_value < self.value_threshold:
                    continue
                implied = quote.implied_probability()
                kelly = KellyCriterion.fraction(adj_probability.win, adj_probability.loss, quote.american_odds)
                opportunities.append(
                    Opportunity(
                        event_id=quote.event_id,
                        sportsbook=quote.sportsbook,
                        book_market_group=quote.book_market_group,
                        market=quote.market,
                        scope=quote.scope,
                        entity_type=quote.entity_type,
                        team_or_player=quote.team_or_player,
                        side=quote.side,
                        line=quote.line,
                        american_odds=quote.american_odds,
                        model_probability=adj_probability.win,
                        push_probability=adj_probability.push,
                        implied_probability=implied,
                        expected_value=expected_value,
                        kelly_fraction=kelly,
                        extra=dict(quote.extra or {}),
                    )
                )
        logger.info("Detected %d potential edges", len(opportunities))
        if self.alert_manager:
            self.alert_manager.notify_edges(
                [
                    {
                        "team_or_player": opp.team_or_player,
                        "american_odds": opp.american_odds,
                        "expected_value": opp.expected_value,
                    }
                    for opp in opportunities
                ]
            )
        return opportunities

    def _probability_for_quote(
        self, quote: OddsQuote, result: SimulationResult | None
    ) -> ProbabilityTriple | None:
        if quote.entity_type == "team" and result:
            if quote.market in {"moneyline", "winner"}:
                win = result.moneyline_probability(quote.team_or_player)
                return ProbabilityTriple(win)
            if quote.market.startswith("spread"):
                if quote.line is None:
                    return None
                return result.spread_probability(
                    quote.team_or_player, quote.line, quote.scope
                )
            if quote.market.startswith("team_total"):
                if quote.line is None or not quote.side:
                    return None
                return result.team_total_probability(
                    quote.team_or_player, quote.side, quote.line, quote.scope
                )
            if quote.market in {"winner_3_way", "moneyline_3_way"}:
                if quote.team_or_player.lower() == "tie":
                    return ProbabilityTriple(result.tie_probability())
                win = result.moneyline_probability(quote.team_or_player)
                return ProbabilityTriple(win)
        if quote.entity_type in {"total", "game_total"} and result:
            if quote.line is None or not quote.side:
                return None
            return result.total_probability(quote.side, quote.line, quote.scope)
        if quote.entity_type == "player":
            return self.player_model.probability(
                quote.team_or_player,
                quote.market,
                quote.side or "over",
                quote.line,
                quote.scope,
                quote.extra or {},
            )
        if quote.entity_type == "leader":
            participants = _extract_participants(quote.extra)
            if quote.team_or_player not in participants:
                participants.append(quote.team_or_player)
            if not participants:
                return None
            scores: Dict[str, Tuple[float, float]] = {}
            for participant in participants:
                mean, stdev = self.player_model.projection_stats(
                    participant,
                    quote.market,
                    quote.scope,
                    quote.extra or {},
                )
                scores[participant] = (mean, stdev)
            baseline = scores.get(quote.team_or_player)
            if baseline is None:
                return None
            numerator = _leader_score(*baseline)
            denominator = 0.0
            for candidate in scores.values():
                denominator += _leader_score(*candidate)
            if denominator <= 0.0:
                return ProbabilityTriple(0.0)
            win = max(0.0, min(1.0, numerator / denominator))
            return ProbabilityTriple(win)
        if quote.entity_type == "either":
            participants = _extract_participants(quote.extra)
            if not participants or quote.line is None:
                return None
            composite_extra: Dict[str, object] = dict(quote.extra or {})
            composite_extra["components"] = participants
            composite_extra["composite_mode"] = "either"
            base_features = composite_extra.get("component_features")
            participant_features: Dict[str, Mapping[str, object]] = {}
            for participant in participants:
                features = dict(base_features.get(participant, composite_extra) if isinstance(base_features, Mapping) else composite_extra)
                features.pop("components", None)
                features.pop("composite_mode", None)
                features.pop("participants", None)
                participant_features[participant] = features
            composite_extra["component_features"] = participant_features
            side = quote.side or "yes"
            return self.player_model.probability(
                quote.team_or_player,
                quote.market,
                side,
                quote.line,
                quote.scope,
                composite_extra,
            )
        return None

    def _evaluate_probabilities(
        self,
        evaluations: Sequence[Tuple[OddsQuote, ProbabilityTriple, SimulationResult | None]],
    ) -> Tuple[List[ProbabilityTriple], List[float]]:
        if self._backend == "numpy" and self._np is not None and len(evaluations) > 1:
            return self._evaluate_probabilities_numpy(evaluations)
        return self._evaluate_probabilities_python(evaluations)

    def _evaluate_probabilities_python(
        self,
        evaluations: Sequence[Tuple[OddsQuote, ProbabilityTriple, SimulationResult | None]],
    ) -> Tuple[List[ProbabilityTriple], List[float]]:
        adjusted: List[ProbabilityTriple] = []
        expected: List[float] = []
        for quote, probability, result in evaluations:
            adj_probability = self._apply_correlation(probability, quote, result)
            adjusted.append(adj_probability)
            multiplier = quote.decimal_multiplier()
            expected.append(adj_probability.win * multiplier - adj_probability.loss)
        return adjusted, expected

    def _evaluate_probabilities_numpy(
        self,
        evaluations: Sequence[Tuple[OddsQuote, ProbabilityTriple, SimulationResult | None]],
    ) -> Tuple[List[ProbabilityTriple], List[float]]:
        np = self._np
        assert np is not None
        wins = np.array([prob.win for _, prob, _ in evaluations], dtype=float)
        pushes = np.array([prob.push for _, prob, _ in evaluations], dtype=float)
        multipliers = np.array([quote.decimal_multiplier() for quote, _, _ in evaluations], dtype=float)
        adjustments = np.array(
            [self._correlation_factor(quote, prob, result) for quote, prob, result in evaluations],
            dtype=float,
        )
        adjusted_wins = wins + adjustments
        adjusted_wins = np.clip(adjusted_wins, 0.0, 1.0 - pushes)
        losses = 1.0 - adjusted_wins - pushes
        expected = adjusted_wins * multipliers - losses
        adjusted = [
            ProbabilityTriple(float(win), float(push))
            for win, push in zip(adjusted_wins.tolist(), pushes.tolist())
        ]
        return adjusted, expected.tolist()

    def _apply_correlation(
        self,
        probability: ProbabilityTriple,
        quote: OddsQuote,
        result: SimulationResult | None,
    ) -> ProbabilityTriple:
        adjustment = self._correlation_factor(quote, probability, result)
        if adjustment == 0.0:
            return probability
        win = max(0.0, min(1.0 - probability.push, probability.win + adjustment))
        return ProbabilityTriple(win, probability.push)

    def _correlation_factor(
        self,
        quote: OddsQuote,
        probability: ProbabilityTriple,
        result: SimulationResult | None,
    ) -> float:
        del probability
        if self.correlation_penalty <= 0.0 or result is None:
            return 0.0
        corr = 0.0
        if quote.extra and "correlation_key" in quote.extra:
            key = (str(quote.extra["correlation_key"]), "")
            reference = (str(quote.extra.get("correlates_with", "total")), "")
            corr = result.correlation(key, reference)
        elif quote.entity_type == "team" and quote.market.startswith("team_total"):
            base = "home_score" if quote.team_or_player == result.home_team else "away_score"
            corr = result.correlation((base, ""), ("total", ""))
        elif quote.entity_type in {"total", "game_total"}:
            corr = result.correlation(("total", ""), ("total", ""))
        else:
            corr = 0.0
        if corr == 0.0:
            return 0.0
        skew = corr * self.correlation_penalty
        if quote.side in {"under", "no"}:
            skew *= -1.0
        return skew

    def benchmark(
        self,
        odds: Sequence[OddsQuote],
        simulations: Sequence[SimulationResult],
        repeats: int = 5,
    ) -> SimulationBenchmark:
        start = time.perf_counter()
        evaluations = 0
        for _ in range(repeats):
            evaluations += len(self.detect(odds, simulations))
        elapsed = max(1e-6, time.perf_counter() - start)
        per_minute = evaluations / (elapsed / 60.0)
        if per_minute < 1_000_000:
            logger.debug(
                "Benchmark throughput %.0f edges/minute below million threshold; consider numpy backend",
                per_minute,
            )
        return SimulationBenchmark(backend=self._backend, simulations_run=evaluations, elapsed_seconds=elapsed)


def _leader_score(mean: float, stdev: float) -> float:
    stdev = max(1e-6, stdev)
    z = mean / stdev
    return max(1e-6, 1.0 + z)


def _extract_participants(extra: Mapping[str, object] | None) -> List[str]:
    if not extra:
        return []
    raw_participants = extra.get("participants")
    if isinstance(raw_participants, Iterable) and not isinstance(raw_participants, (str, bytes)):
        return [
            participant
            for participant in raw_participants
            if isinstance(participant, str)
        ]
    return []


@dataclasses.dataclass(slots=True)
class LineMovement:
    key: Tuple[str, str, str, str, str | None, float | None]
    opening_price: int
    latest_price: int
    opening_time: dt.datetime
    latest_time: dt.datetime

    @property
    def delta(self) -> int:
        return self.latest_price - self.opening_price


class LineMovementAnalyzer:
    """Summarise line movement using stored odds history."""

    def __init__(
        self,
        history: Sequence["IngestedOdds"],
        *,
        alert_manager: AlertManager | None = None,
        alert_threshold: int = 40,
    ) -> None:
        self.history = list(history)
        self.alert_manager = alert_manager or get_alert_manager()
        self.alert_threshold = alert_threshold

    def summarise(self) -> List[LineMovement]:
        grouped: Dict[
            Tuple[str, str, str, str, str | None, float | None],
            List["IngestedOdds"],
        ] = collections.defaultdict(list)
        for row in self.history:
            key = (
                row.event_id,
                row.market,
                row.scope,
                row.team_or_player,
                row.side,
                row.line,
            )
            grouped[key].append(row)
        movements: List[LineMovement] = []
        for key, rows in grouped.items():
            ordered_rows = sorted(rows, key=lambda row: row.observed_at)
            if len(ordered_rows) < 2:
                continue
            opening = ordered_rows[0]
            latest = ordered_rows[-1]
            movements.append(
                LineMovement(
                    key=key,
                    opening_price=opening.american_odds,
                    latest_price=latest.american_odds,
                    opening_time=opening.observed_at,
                    latest_time=latest.observed_at,
                )
            )
        ordered_movements = sorted(
            movements, key=lambda movement: abs(movement.delta), reverse=True
        )
        if self.alert_manager:
            self.alert_manager.notify_line_movement(
                [
                    {
                        "key": movement.key,
                        "delta": movement.delta,
                    }
                    for movement in ordered_movements
                ],
                threshold=self.alert_threshold,
            )
        return ordered_movements


@dataclasses.dataclass(slots=True)
class PortfolioPosition:
    opportunity: Opportunity
    stake: float


@dataclasses.dataclass(slots=True)
class BankrollTrajectory:
    balances: List[float]

    @property
    def terminal_balance(self) -> float:
        return self.balances[-1]

    @property
    def max_drawdown(self) -> float:
        peak = -math.inf
        max_dd = 0.0
        for balance in self.balances:
            peak = max(peak, balance) if peak != -math.inf else balance
            if peak <= 0:
                continue
            drawdown = (peak - balance) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd


@dataclasses.dataclass(slots=True)
class BankrollSimulationResult:
    trajectories: List[BankrollTrajectory]

    @property
    def trials(self) -> int:
        return len(self.trajectories)

    @property
    def terminal_balances(self) -> List[float]:
        return [trajectory.terminal_balance for trajectory in self.trajectories]

    @property
    def max_drawdowns(self) -> List[float]:
        return [trajectory.max_drawdown for trajectory in self.trajectories]

    def summary(self) -> Mapping[str, float]:
        if not self.trajectories:
            return {
                "trials": 0.0,
                "mean_terminal": 0.0,
                "median_terminal": 0.0,
                "worst_terminal": 0.0,
                "average_drawdown": 0.0,
                "worst_drawdown": 0.0,
                "p05_drawdown": 0.0,
                "p95_drawdown": 0.0,
            }
        terminals = self.terminal_balances
        drawdowns = self.max_drawdowns
        return {
            "trials": float(self.trials),
            "mean_terminal": statistics.mean(terminals),
            "median_terminal": statistics.median(terminals),
            "worst_terminal": min(terminals),
            "average_drawdown": statistics.mean(drawdowns) if drawdowns else 0.0,
            "worst_drawdown": max(drawdowns) if drawdowns else 0.0,
            "p05_drawdown": _percentile(drawdowns, 0.05),
            "p95_drawdown": _percentile(drawdowns, 0.95),
        }


def _percentile(values: Iterable[float], percentile: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    if percentile <= 0:
        return data[0]
    if percentile >= 1:
        return data[-1]
    index = (len(data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return data[int(index)]
    lower_value = data[lower]
    upper_value = data[upper]
    weight = index - lower
    return lower_value + (upper_value - lower_value) * weight


class PortfolioManager:
    """Risk-aware staking engine with exposure caps."""

    def __init__(
        self,
        bankroll: float,
        max_risk_per_bet: float = 0.02,
        max_event_exposure: float = 0.1,
        compliance_engine: ComplianceEngine | None = None,
        responsible_gaming: ResponsibleGamingControls | None = None,
        audit_logger: logging.Logger | None = None,
        *,
        fractional_kelly: float = 1.0,
        correlation_limits: Mapping[str, float] | None = None,
        correlation_key: Callable[[Opportunity], str | None] | None = None,
    ) -> None:
        self.bankroll = bankroll
        self.max_risk_per_bet = max_risk_per_bet
        self.max_event_exposure = max_event_exposure
        self.positions: List[PortfolioPosition] = []
        self._exposure: Dict[Tuple[str, str], float] = collections.defaultdict(float)
        self._correlated_exposure: collections.defaultdict[str, float] = collections.defaultdict(float)
        self._compliance_engine = compliance_engine
        self._controls = responsible_gaming or ResponsibleGamingControls()
        self._audit_logger = audit_logger or logging.getLogger("nflreadpy.betting.audit")
        self._session_start_bankroll = bankroll
        self._total_session_stake = 0.0
        self._cooldown_until: dt.datetime | None = None
        self.fractional_kelly = max(0.0, fractional_kelly)
        self._correlation_limits = dict(correlation_limits or {})
        self._correlation_key = correlation_key or self._default_correlation_key

    def allocate(self, opportunity: Opportunity) -> PortfolioPosition | None:
        now = dt.datetime.now(dt.timezone.utc)
        if self._cooldown_until and now < self._cooldown_until:
            self._audit_logger.warning(
                "portfolio.rejected",
                extra={
                    "reason": "cooldown_active",
                    "event_id": opportunity.event_id,
                    "market": opportunity.market,
                    "sportsbook": opportunity.sportsbook,
                },
            )
            return None

        if self._compliance_engine and not self._compliance_engine.validate(opportunity):
            self._audit_logger.warning(
                "portfolio.rejected",
                extra={
                    "reason": "compliance",
                    "event_id": opportunity.event_id,
                    "market": opportunity.market,
                    "sportsbook": opportunity.sportsbook,
                },
            )
            return None

        session_loss = self._session_start_bankroll - self.bankroll
        if (
            self._controls.session_loss_limit is not None
            and session_loss >= self._controls.session_loss_limit
        ):
            self._start_cooldown(now)
            self._audit_logger.warning(
                "portfolio.rejected",
                extra={
                    "reason": "session_loss_limit",
                    "event_id": opportunity.event_id,
                    "market": opportunity.market,
                    "sportsbook": opportunity.sportsbook,
                },
            )
            return None

        raw_stake = self.bankroll * self.max_risk_per_bet
        effective_fraction = max(0.0, opportunity.kelly_fraction) * self.fractional_kelly
        kelly_stake = self.bankroll * effective_fraction
        stake = min(raw_stake, kelly_stake)
        if stake <= 0.0:
            return None
        event_key = (opportunity.event_id, opportunity.market)
        cap = self.bankroll * self.max_event_exposure
        remaining = cap - self._exposure[event_key]
        if remaining <= 0.0:
            return None
        stake = min(stake, remaining)
        correlation_key = self._correlation_key(opportunity)
        if correlation_key is not None:
            limit_fraction = self._correlation_limits.get(correlation_key)
            if limit_fraction is not None:
                correlation_cap = self.bankroll * limit_fraction
                remaining_corr = correlation_cap - self._correlated_exposure[correlation_key]
                if remaining_corr <= 0.0:
                    return None
                stake = min(stake, remaining_corr)
        if self._controls.session_stake_limit is not None:
            remaining_session = self._controls.session_stake_limit - self._total_session_stake
            if remaining_session <= 0:
                self._start_cooldown(now)
                self._audit_logger.warning(
                    "portfolio.rejected",
                    extra={
                        "reason": "session_stake_limit",
                        "event_id": opportunity.event_id,
                        "market": opportunity.market,
                        "sportsbook": opportunity.sportsbook,
                    },
                )
                return None
            stake = min(stake, remaining_session)
        if stake <= 0.0:
            return None
        self.bankroll -= stake
        self._exposure[event_key] += stake
        if correlation_key is not None:
            self._correlated_exposure[correlation_key] += stake
        position = PortfolioPosition(opportunity=opportunity, stake=stake)
        self.positions.append(position)
        self._total_session_stake += stake
        self._audit_logger.info(
            "portfolio.accepted",
            extra={
                "event_id": opportunity.event_id,
                "market": opportunity.market,
                "sportsbook": opportunity.sportsbook,
                "stake": stake,
            },
        )
        return position

    def exposure_report(self) -> Dict[Tuple[str, str], float]:
        return dict(self._exposure)

    def correlation_report(self) -> Dict[str, float]:
        return dict(self._correlated_exposure)

    def reset_session(self, bankroll: float | None = None) -> None:
        """Reset session counters and optionally bankroll."""

        if bankroll is not None:
            self.bankroll = bankroll
        self._session_start_bankroll = self.bankroll
        self._total_session_stake = 0.0
        self._cooldown_until = None

    def simulate_bankroll(
        self,
        *,
        trials: int = 500,
        seed: int | None = None,
        positions: Sequence[PortfolioPosition] | None = None,
    ) -> BankrollSimulationResult:
        if trials <= 0:
            raise ValueError("Simulation trials must be positive")
        sample_positions = list(positions or self.positions)
        rng = random.Random(seed)
        trajectories: List[BankrollTrajectory] = []
        start = self._session_start_bankroll
        for _ in range(trials):
            bankroll = start
            balances = [bankroll]
            for position in sample_positions:
                opportunity = position.opportunity
                payout = opportunity.decimal_multiplier()
                win_prob = max(0.0, min(1.0, opportunity.model_probability))
                push_prob = max(0.0, min(1.0 - win_prob, opportunity.push_probability))
                roll = rng.random()
                if roll < win_prob:
                    bankroll += position.stake * payout
                elif roll < win_prob + push_prob:
                    bankroll += 0.0
                else:
                    bankroll -= position.stake
                balances.append(bankroll)
            trajectories.append(BankrollTrajectory(balances))
        return BankrollSimulationResult(trajectories)

    def _start_cooldown(self, reference: dt.datetime) -> None:
        if self._controls.cooldown_seconds:
            self._cooldown_until = reference + dt.timedelta(
                seconds=self._controls.cooldown_seconds
            )

    @staticmethod
    def _default_correlation_key(opportunity: Opportunity) -> str | None:
        if opportunity.extra and isinstance(opportunity.extra, Mapping):
            value = opportunity.extra.get("correlation_group")
            if isinstance(value, str):
                return value
        return None


def consolidate_best_prices(opportunities: Sequence[Opportunity]) -> List[Opportunity]:
    """Filter opportunities down to the best available price per selection."""

    if not opportunities:
        return []
    quotes = [
        OddsQuote(
            event_id=opp.event_id,
            sportsbook=opp.sportsbook,
            book_market_group=opp.book_market_group,
            market=opp.market,
            scope=opp.scope,
            entity_type=opp.entity_type,
            team_or_player=opp.team_or_player,
            side=opp.side,
            line=opp.line,
            american_odds=opp.american_odds,
            observed_at=dt.datetime.now(dt.timezone.utc),
            extra=opp.extra,
        )
        for opp in opportunities
    ]
    best = best_prices_by_selection(quotes)
    filtered: List[Opportunity] = []
    for opportunity in opportunities:
        key = (
            opportunity.event_id,
            opportunity.market,
            opportunity.scope,
            opportunity.team_or_player,
            opportunity.side,
            opportunity.line,
        )
        best_quote = best.get(key)
        if best_quote and best_quote.american_odds == opportunity.american_odds:
            filtered.append(opportunity)
    return filtered

