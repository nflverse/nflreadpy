"""Shared state containers and helpers for betting dashboards."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Tuple

from .analytics import (
    BankrollSimulationResult,
    Opportunity,
    PortfolioPosition,
    OptimizerComparisonResult,
)
from .ingestion import IngestedOdds
from .models import SimulationResult


@dataclasses.dataclass(slots=True, frozen=True)
class LadderCell:
    """Display payload for ladder matrices."""

    american_odds: int
    probability: float | None = None
    raw_probability: float | None = None

    def summary(self, *, precision: int = 1) -> str:
        base = f"{self.american_odds:+}"
        if self.probability is None:
            return base
        pct = f"{self.probability:.{precision}%}"
        return f"{base} {pct}"

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "american_odds": self.american_odds,
            "probability": self.probability,
            "raw_probability": self.raw_probability,
        }

DEFAULT_SEARCH_TARGETS = frozenset({"quotes", "opportunities", "simulations"})


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardPanelState:
    """Represents whether a panel is visible to the user."""

    key: str
    title: str
    collapsed: bool = False


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardSearchState:
    """Metadata describing the active dashboard search query."""

    query: str | None = None
    case_sensitive: bool = False
    targets: frozenset[str] = dataclasses.field(default_factory=lambda: DEFAULT_SEARCH_TARGETS)

    def normalised_query(self) -> str | None:
        if self.query is None:
            return None
        return self.query if self.case_sensitive else self.query.lower()

    def update(
        self,
        *,
        query: str | None | object = dataclasses.MISSING,
        case_sensitive: bool | object = dataclasses.MISSING,
        targets: Iterable[str] | object = dataclasses.MISSING,
    ) -> "DashboardSearchState":
        payload: dict[str, object] = {}
        if query is not dataclasses.MISSING:
            payload["query"] = query
        if case_sensitive is not dataclasses.MISSING:
            payload["case_sensitive"] = bool(case_sensitive)
        if targets is not dataclasses.MISSING:
            payload["targets"] = DEFAULT_SEARCH_TARGETS if targets is None else frozenset(targets)
        return dataclasses.replace(self, **payload)

    def match(self, text: str) -> bool:
        """Return ``True`` when ``text`` satisfies the current query."""

        if not self.query:
            return True
        haystack = text if self.case_sensitive else text.lower()
        needle = self.normalised_query()
        return bool(needle and needle in haystack)


@dataclasses.dataclass(slots=True)
class RiskSummary:
    """Aggregate bankroll and exposure metrics for the risk panel."""

    bankroll: float
    opportunity_fraction: float
    portfolio_fraction: float
    positions: Sequence[PortfolioPosition] = dataclasses.field(default_factory=tuple)
    exposure_by_event: Mapping[Tuple[str, str], float] = dataclasses.field(default_factory=dict)
    correlation_exposure: Mapping[str, float] = dataclasses.field(default_factory=dict)
    correlation_limits: Mapping[str, float] = dataclasses.field(default_factory=dict)
    simulation: BankrollSimulationResult | None = None
    bankroll_summary: Mapping[str, float] | None = None
    optimizer_comparison: OptimizerComparisonResult | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardFilters:
    """Collection of filters applied to dashboard data."""

    sportsbooks: frozenset[str] | None = None
    market_groups: frozenset[str] | None = None
    markets: frozenset[str] | None = None
    scopes: frozenset[str] | None = None
    events: frozenset[str] | None = None
    include_quarters: bool = True
    include_halves: bool = True

    def update(
        self,
        *,
        sportsbooks: Iterable[str] | None | object = dataclasses.MISSING,
        market_groups: Iterable[str] | None | object = dataclasses.MISSING,
        markets: Iterable[str] | None | object = dataclasses.MISSING,
        scopes: Iterable[str] | None | object = dataclasses.MISSING,
        events: Iterable[str] | None | object = dataclasses.MISSING,
        include_quarters: bool | object = dataclasses.MISSING,
        include_halves: bool | object = dataclasses.MISSING,
    ) -> "DashboardFilters":
        """Return a new instance with provided updates."""

        payload: dict[str, object] = {}
        if sportsbooks is not dataclasses.MISSING:
            payload["sportsbooks"] = freeze_optional(sportsbooks)
        if market_groups is not dataclasses.MISSING:
            payload["market_groups"] = freeze_optional(market_groups)
        if markets is not dataclasses.MISSING:
            payload["markets"] = freeze_optional(markets)
        if scopes is not dataclasses.MISSING:
            normalized = None
            if scopes is not None:
                normalized = frozenset(normalize_scope(scope) for scope in scopes)
            payload["scopes"] = normalized
        if events is not dataclasses.MISSING:
            payload["events"] = freeze_optional(events)
        if include_quarters is not dataclasses.MISSING:
            payload["include_quarters"] = bool(include_quarters)
        if include_halves is not dataclasses.MISSING:
            payload["include_halves"] = bool(include_halves)
        return dataclasses.replace(self, **payload)

    def match_odds(self, quote: IngestedOdds) -> bool:
        """Return ``True`` if the odds quote passes active filters."""

        if self.sportsbooks and quote.sportsbook not in self.sportsbooks:
            return False
        if self.market_groups and quote.book_market_group not in self.market_groups:
            return False
        if self.markets and quote.market not in self.markets:
            return False
        scope = normalize_scope(quote.scope)
        if not self.include_quarters and is_quarter_scope(scope):
            return False
        if not self.include_halves and is_half_scope(scope):
            return False
        if self.scopes and scope not in self.scopes:
            return False
        if self.events and quote.event_id not in self.events:
            return False
        return True

    def match_opportunity(self, opportunity: Opportunity) -> bool:
        if self.sportsbooks and opportunity.sportsbook not in self.sportsbooks:
            return False
        if self.market_groups and opportunity.book_market_group not in self.market_groups:
            return False
        if self.markets and opportunity.market not in self.markets:
            return False
        scope = normalize_scope(opportunity.scope)
        if not self.include_quarters and is_quarter_scope(scope):
            return False
        if not self.include_halves and is_half_scope(scope):
            return False
        if self.scopes and scope not in self.scopes:
            return False
        if self.events and opportunity.event_id not in self.events:
            return False
        return True

    def match_simulation(self, result: SimulationResult) -> bool:
        if self.events and result.event_id not in self.events:
            return False
        return True

    def description(self) -> list[str]:
        """Return human readable summary of active filters."""

        pieces: list[str] = []
        pieces.append(describe_filter_dimension("Sportsbooks", self.sportsbooks))
        pieces.append(describe_filter_dimension("Market groups", self.market_groups))
        pieces.append(describe_filter_dimension("Markets", self.markets))
        pieces.append(describe_filter_dimension("Scopes", self.scopes))
        pieces.append(describe_filter_dimension("Events", self.events))
        pieces.append(
            f"Quarters={'on' if self.include_quarters else 'off'} | "
            f"Halves={'on' if self.include_halves else 'off'}"
        )
        return pieces


@dataclasses.dataclass(slots=True)
class DashboardContext:
    filters: DashboardFilters
    search: DashboardSearchState
    odds: Sequence[IngestedOdds]
    simulations: Sequence[SimulationResult]
    opportunities: Sequence[Opportunity]
    search_results: dict[str, Sequence[object]]
    risk_summary: RiskSummary | None = None
    movement_summary: Sequence[LineMovement] = dataclasses.field(default_factory=tuple)
    movement_series: Mapping[
        Tuple[str, str, str, str, str | None, float | None],
        Tuple[Tuple[dt.datetime, int], ...],
    ] = dataclasses.field(default_factory=dict)
    movement_sparklines: Mapping[
        Tuple[str, str, str, str, str | None, float | None],
        str,
    ] = dataclasses.field(default_factory=dict)
    movement_depth: int | None = None
    movement_threshold: int | None = None
    stale_after: dt.timedelta | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardPanelView:
    state: DashboardPanelState
    body: tuple[str, ...]


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardSnapshot:
    header: tuple[str, ...]
    panels: tuple[DashboardPanelView, ...]
    context: DashboardContext


def freeze_optional(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    frozen = frozenset(values)
    return frozen or None


def normalize_scope(scope: str) -> str:
    return scope.strip().lower().replace(" ", "")


def is_quarter_scope(scope: str) -> bool:
    normalized = scope.replace(" ", "").lower()
    canonical = normalized.replace("quarter", "q")
    return canonical in {
        "q1",
        "1q",
        "1stq",
        "firstq",
        "q2",
        "2q",
        "2ndq",
        "secondq",
        "q3",
        "3q",
        "3rdq",
        "thirdq",
        "q4",
        "4q",
        "4thq",
        "fourthq",
    }


def is_half_scope(scope: str) -> bool:
    normalized = scope.replace(" ", "").lower()
    canonical = normalized.replace("half", "h")
    return canonical in {
        "h1",
        "1h",
        "1sth",
        "firsth",
        "fh",
        "h2",
        "2h",
        "2ndh",
        "secondh",
        "sh",
    } or normalized in {"firsthalf", "secondhalf", "1sthalf", "2ndhalf"}


def describe_filter_dimension(name: str, values: frozenset[str] | None) -> str:
    if not values:
        return f"{name}: All"
    return f"{name}: {', '.join(sorted(values))}"


def build_ladder_matrix(
    odds: Sequence[IngestedOdds],
) -> dict[tuple[str, str, str], dict[str, dict[float, LadderCell]]]:
    ladder: dict[tuple[str, str, str], dict[str, dict[float, LadderCell]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for quote in odds:
        if quote.line is None:
            continue
        key = (quote.event_id, quote.market, quote.team_or_player)
        scope = normalize_scope(quote.scope)
        rounded_line = round(float(quote.line), 1)
        tail_info = _tail_probability_payload(quote.extra)
        probability = (
            float(tail_info.get("final_win"))
            if tail_info and isinstance(tail_info.get("final_win"), (int, float))
            else (
                float(tail_info.get("adjusted_win"))
                if tail_info and isinstance(tail_info.get("adjusted_win"), (int, float))
                else None
            )
        )
        raw_probability = None
        if tail_info and isinstance(tail_info.get("raw_win"), (int, float)):
            raw_probability = float(tail_info["raw_win"])
        ladder[key][scope][rounded_line] = LadderCell(
            american_odds=quote.american_odds,
            probability=probability,
            raw_probability=raw_probability,
        )
    return {k: dict(v) for k, v in ladder.items()}


def _tail_probability_payload(extra: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if not isinstance(extra, Mapping):
        return None
    tail = extra.get("tail_probability")
    return tail if isinstance(tail, Mapping) else None


def parse_filter_tokens(tokens: Sequence[str]) -> dict[str, object]:
    updates: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid filter token: {token}")
        key, value = token.split("=", 1)
        key = key.strip()
        values = [item.strip() for item in value.split(",") if item.strip()]
        updates[key] = None if not values else values
    return updates


def movement_key(row: IngestedOdds | LineMovement) -> Tuple[str, str, str, str, str | None, float | None]:
    """Return the grouping key for movement analytics."""

    if isinstance(row, LineMovement):
        return row.key
    return (
        row.event_id,
        row.market,
        row.scope,
        row.team_or_player,
        row.side,
        row.line,
    )


def build_movement_series(
    history: Sequence[IngestedOdds],
    *,
    max_points: int | None = None,
) -> dict[
    Tuple[str, str, str, str, str | None, float | None],
    tuple[tuple[dt.datetime, int], ...],
]:
    """Group odds history into ordered time-series per selection."""

    grouped: defaultdict[
        Tuple[str, str, str, str, str | None, float | None],
        list[tuple[dt.datetime, int]],
    ] = defaultdict(list)
    for row in history:
        key = movement_key(row)
        grouped[key].append((row.observed_at, row.american_odds))
    result: dict[
        Tuple[str, str, str, str, str | None, float | None],
        tuple[tuple[dt.datetime, int], ...],
    ] = {}
    for key, entries in grouped.items():
        ordered = sorted(entries, key=lambda item: item[0])
        if max_points is not None and max_points >= 0:
            ordered = ordered[-max_points:]
        result[key] = tuple(ordered)
    return result


_SPARKLINE_STEPS = "▁▂▃▄▅▆▇█"


def render_sparkline(values: Sequence[int]) -> str:
    """Render ``values`` as a simple unicode sparkline."""

    if not values:
        return ""
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return _SPARKLINE_STEPS[len(_SPARKLINE_STEPS) // 2] * len(values)
    scale = maximum - minimum
    return "".join(
        _SPARKLINE_STEPS[
            int(round(((value - minimum) / scale) * (len(_SPARKLINE_STEPS) - 1)))
        ]
        for value in values
    )


def format_age(delta: dt.timedelta) -> str:
    """Return a compact representation of ``delta`` for freshness indicators."""

    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


__all__ = [
    "DEFAULT_SEARCH_TARGETS",
    "DashboardContext",
    "DashboardFilters",
    "LadderCell",
    "DashboardPanelState",
    "DashboardPanelView",
    "DashboardSearchState",
    "DashboardSnapshot",
    "RiskSummary",
    "build_ladder_matrix",
    "build_movement_series",
    "describe_filter_dimension",
    "freeze_optional",
    "format_age",
    "is_half_scope",
    "is_quarter_scope",
    "movement_key",
    "normalize_scope",
    "parse_filter_tokens",
    "render_sparkline",
]
