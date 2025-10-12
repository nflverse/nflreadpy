"""Edge detection and bankroll sizing utilities."""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import logging
import time
from typing import Dict, List, Mapping, Sequence, Tuple

from .alerts import AlertManager, get_alert_manager
from .compliance import ResponsibleGamingControls
from .models import (
    PlayerPropForecaster,
    ProbabilityTriple,
    SimulationBenchmark,
    SimulationResult,
)
from .scrapers.base import OddsQuote, best_prices_by_selection

try:  # pragma: no cover - optional import for type checking
    from .ingestion import IngestedOdds
except Exception:  # pragma: no cover
    IngestedOdds = object  # type: ignore

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class Opportunity:
    event_id: str
    sportsbook: str
    book_market_group: str
    market: str
    scope: str
    entity_type: str
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


class KellyCriterion:
    """Utility for computing fractional Kelly bet sizes."""

    @staticmethod
    def fraction(win_probability: float, loss_probability: float, price: int) -> float:
        if price == 0:
            raise ValueError("American odds cannot be zero")
        b = price / 100.0 if price > 0 else 100.0 / -price
        numerator = b * win_probability - loss_probability
        if numerator <= 0:
            return 0.0
        return numerator / b


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
                import numpy as np  # type: ignore

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
            participants = list(
                quote.extra.get("participants", [])  # type: ignore[assignment]
                if quote.extra
                else []
            )
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
            participants = list(
                quote.extra.get("participants", [])  # type: ignore[assignment]
                if quote.extra
                else []
            )
            if not participants or quote.line is None:
                return None
            probs = [
                self.player_model.probability(
                    participant,
                    quote.market,
                    quote.side or "over",
                    quote.line,
                    quote.scope,
                    quote.extra or {},
                ).win
                for participant in participants
            ]
            win = 1.0
            for prob in probs:
                win *= 1.0 - prob
            win = 1.0 - win
            return ProbabilityTriple(max(0.0, min(1.0, win)))
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
            ordered = sorted(rows, key=lambda row: row.observed_at)
            if len(ordered) < 2:
                continue
            opening = ordered[0]
            latest = ordered[-1]
            movements.append(
                LineMovement(
                    key=key,
                    opening_price=opening.american_odds,
                    latest_price=latest.american_odds,
                    opening_time=opening.observed_at,
                    latest_time=latest.observed_at,
                )
            )
        ordered = sorted(movements, key=lambda movement: abs(movement.delta), reverse=True)
        if self.alert_manager:
            self.alert_manager.notify_line_movement(
                [
                    {
                        "key": movement.key,
                        "delta": movement.delta,
                    }
                    for movement in ordered
                ],
                threshold=self.alert_threshold,
            )
        return ordered


@dataclasses.dataclass(slots=True)
class PortfolioPosition:
    opportunity: Opportunity
    stake: float


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
    ) -> None:
        self.bankroll = bankroll
        self.max_risk_per_bet = max_risk_per_bet
        self.max_event_exposure = max_event_exposure
        self.positions: List[PortfolioPosition] = []
        self._exposure: Dict[Tuple[str, str], float] = collections.defaultdict(float)
        self._compliance_engine = compliance_engine
        self._controls = responsible_gaming or ResponsibleGamingControls()
        self._audit_logger = audit_logger or logging.getLogger("nflreadpy.betting.audit")
        self._session_start_bankroll = bankroll
        self._total_session_stake = 0.0
        self._cooldown_until: dt.datetime | None = None

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
        kelly_stake = self.bankroll * max(0.0, opportunity.kelly_fraction)
        stake = min(raw_stake, kelly_stake)
        if stake <= 0.0:
            return None
        event_key = (opportunity.event_id, opportunity.market)
        cap = self.bankroll * self.max_event_exposure
        remaining = cap - self._exposure[event_key]
        if remaining <= 0.0:
            return None
        stake = min(stake, remaining)
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

    def reset_session(self, bankroll: float | None = None) -> None:
        """Reset session counters and optionally bankroll."""

        if bankroll is not None:
            self.bankroll = bankroll
        self._session_start_bankroll = self.bankroll
        self._total_session_stake = 0.0
        self._cooldown_until = None

    def _start_cooldown(self, reference: dt.datetime) -> None:
        if self._controls.cooldown_seconds:
            self._cooldown_until = reference + dt.timedelta(
                seconds=self._controls.cooldown_seconds
            )


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

