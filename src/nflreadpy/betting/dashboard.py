"""ASCII dashboard inspired by Bloomberg-style terminal layouts.

This module implements an interactive console dashboard with filter controls,
panel toggles, and ladder matrix summaries.  The same filtering logic is reused
by the Streamlit web dashboard, allowing the two interfaces to stay in sync.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import shlex
from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Tuple

from .analytics import BankrollSimulationResult, Opportunity, PortfolioPosition
from .ingestion import IngestedOdds
from .models import SimulationResult


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
            payload["sportsbooks"] = _freeze_optional(sportsbooks)
        if market_groups is not dataclasses.MISSING:
            payload["market_groups"] = _freeze_optional(market_groups)
        if markets is not dataclasses.MISSING:
            payload["markets"] = _freeze_optional(markets)
        if scopes is not dataclasses.MISSING:
            normalized = None
            if scopes is not None:
                normalized = frozenset(_normalize_scope(scope) for scope in scopes)
            payload["scopes"] = normalized
        if events is not dataclasses.MISSING:
            payload["events"] = _freeze_optional(events)
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
        scope = _normalize_scope(quote.scope)
        if not self.include_quarters and _is_quarter_scope(scope):
            return False
        if not self.include_halves and _is_half_scope(scope):
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
        scope = _normalize_scope(opportunity.scope)
        if not self.include_quarters and _is_quarter_scope(scope):
            return False
        if not self.include_halves and _is_half_scope(scope):
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
        pieces.append(_describe_dimension("Sportsbooks", self.sportsbooks))
        pieces.append(_describe_dimension("Market groups", self.market_groups))
        pieces.append(_describe_dimension("Markets", self.markets))
        pieces.append(_describe_dimension("Scopes", self.scopes))
        pieces.append(_describe_dimension("Events", self.events))
        pieces.append(
            f"Quarters={'on' if self.include_quarters else 'off'} | "
            f"Halves={'on' if self.include_halves else 'off'}"
        )
        return pieces


@dataclasses.dataclass(slots=True)
class DashboardContext:
    """Container for filtered data passed to panel renderers."""

    filters: DashboardFilters
    search: DashboardSearchState
    odds: Sequence[IngestedOdds]
    simulations: Sequence[SimulationResult]
    opportunities: Sequence[Opportunity]
    search_results: dict[str, Sequence[object]]
    risk_summary: "RiskSummary | None"


@dataclasses.dataclass(slots=True, frozen=True)
class RiskSummary:
    """Aggregated view of bankroll exposure for the risk panel."""

    bankroll: float
    opportunity_fraction: float
    portfolio_fraction: float
    positions: Sequence[PortfolioPosition]
    exposure_by_event: Mapping[tuple[str, str], float]
    correlation_exposure: Mapping[str, float]
    simulation: BankrollSimulationResult | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardPanelView:
    """Rendered content for a single dashboard panel."""

    state: DashboardPanelState
    body: tuple[str, ...]


@dataclasses.dataclass(slots=True, frozen=True)
class DashboardSnapshot:
    """Structured representation of the dashboard output."""

    header: tuple[str, ...]
    panels: tuple[DashboardPanelView, ...]
    context: DashboardContext


class Dashboard:
    """Render odds, model outputs, and opportunities in tabular form."""

    def __init__(self) -> None:
        self.filters = DashboardFilters()
        self.search = DashboardSearchState()
        self._panel_order = [
            "controls",
            "simulations",
            "quotes",
            "opportunities",
            "risk",
            "ladders",
            "search",
        ]
        self._panels = {
            "controls": DashboardPanelState("controls", "Controls"),
            "simulations": DashboardPanelState("simulations", "Simulations"),
            "quotes": DashboardPanelState("quotes", "Latest Quotes"),
            "opportunities": DashboardPanelState("opportunities", "Opportunities"),
            "risk": DashboardPanelState("risk", "Risk Management"),
            "ladders": DashboardPanelState("ladders", "Line Ladders"),
            "search": DashboardPanelState("search", "Search Results"),
        }

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def set_filters(self, **kwargs: object) -> None:
        """Update filters with keyword arguments matching :class:`DashboardFilters`."""

        self.filters = self.filters.update(**kwargs)

    def reset_filters(self) -> None:
        """Reset all filters to their default state."""

        self.filters = DashboardFilters()

    def set_search(self, query: str | None, *, case_sensitive: bool | None = None, targets: Iterable[str] | None = None) -> None:
        """Set the active search query across dashboard panels."""

        updates: dict[str, object] = {"query": query}
        if case_sensitive is not None:
            updates["case_sensitive"] = case_sensitive
        if targets is not None:
            updates["targets"] = frozenset(targets)
        self.search = self.search.update(**updates)

    def reset_search(self) -> None:
        """Clear the active search query."""

        self.search = DashboardSearchState()

    def toggle_quarters(self) -> bool:
        """Toggle quarter markets and return the new state."""

        self.filters = self.filters.update(include_quarters=not self.filters.include_quarters)
        return self.filters.include_quarters

    def toggle_halves(self) -> bool:
        """Toggle half markets and return the new state."""

        self.filters = self.filters.update(include_halves=not self.filters.include_halves)
        return self.filters.include_halves

    def toggle_panel(self, key: str) -> bool:
        """Toggle a panel's collapsed state and return whether it is now visible."""

        panel = self._panels.get(key)
        if panel is None:
            raise KeyError(f"Unknown panel '{key}'")
        new_panel = DashboardPanelState(key=panel.key, title=panel.title, collapsed=not panel.collapsed)
        self._panels[key] = new_panel
        return not new_panel.collapsed

    def reorder_panels(self, order: Sequence[str]) -> None:
        """Reorder panels according to the provided keys."""

        missing = set(order) - set(self._panels)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise KeyError(f"Unknown panels in order: {missing_keys}")
        self._panel_order = list(order)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def snapshot(
        self,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
        *,
        risk_summary: RiskSummary | None = None,
    ) -> DashboardSnapshot:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        header = [f"NFL Terminal — {now}", "=" * 96]
        filtered_odds = [quote for quote in odds if self.filters.match_odds(quote)]
        filtered_sims = [result for result in simulations if self.filters.match_simulation(result)]
        filtered_opps = [opp for opp in opportunities if self.filters.match_opportunity(opp)]
        search_hits = self._apply_search(filtered_odds, filtered_sims, filtered_opps)
        context = DashboardContext(
            filters=self.filters,
            search=self.search,
            odds=filtered_odds,
            simulations=filtered_sims,
            opportunities=filtered_opps,
            search_results=search_hits,
            risk_summary=risk_summary,
        )
        panels: list[DashboardPanelView] = []
        for key in self._panel_order:
            panel = self._panels[key]
            panels.append(DashboardPanelView(panel, self._panel_body(panel, context)))
        return DashboardSnapshot(tuple(header), tuple(panels), context)

    def render(
        self,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
        *,
        risk_summary: RiskSummary | None = None,
    ) -> str:
        snapshot = self.snapshot(
            odds,
            simulations,
            opportunities,
            risk_summary=risk_summary,
        )
        sections = ["\n".join(snapshot.header)]
        for view in snapshot.panels:
            panel = view.state
            separator = "-" * len(panel.title)
            if panel.collapsed:
                sections.append(f"{panel.title} (collapsed)\n{separator}")
                continue
            if not view.body:
                sections.append(f"{panel.title}\n{separator}\nNo data available.")
                continue
            sections.append(
                "\n".join((f"{panel.title}", separator, *view.body))
            )
        return "\n".join(section for section in sections if section)

    def available_options(
        self,
        odds: Sequence[IngestedOdds],
        opportunities: Sequence[Opportunity] | None = None,
    ) -> dict[str, list[str]]:
        """Return sorted lists of unique option values for interactive controls."""

        opportunities = opportunities or []
        sportsbooks = {quote.sportsbook for quote in odds}
        sportsbooks.update(opp.sportsbook for opp in opportunities)
        market_groups = {quote.book_market_group for quote in odds}
        market_groups.update(opp.book_market_group for opp in opportunities)
        markets = {quote.market for quote in odds}
        markets.update(opp.market for opp in opportunities)
        scopes = {_normalize_scope(quote.scope) for quote in odds}
        scopes.update(_normalize_scope(opp.scope) for opp in opportunities)
        events = {quote.event_id for quote in odds}
        events.update(opp.event_id for opp in opportunities)
        return {
            "sportsbooks": sorted(sportsbooks),
            "market_groups": sorted(market_groups),
            "markets": sorted(markets),
            "scopes": sorted(scopes),
            "events": sorted(events),
        }

    def _panel_body(self, panel: DashboardPanelState, context: DashboardContext) -> tuple[str, ...]:
        if panel.collapsed:
            return ()
        renderer = getattr(self, f"_render_{panel.key}")
        content = renderer(context)
        if not content:
            return ()
        if isinstance(content, str):
            return tuple(content.splitlines())
        return tuple(content)

    def _render_controls(self, context: DashboardContext) -> str:
        lines = ["Active Filters:"]
        for piece in context.filters.description():
            lines.append(f"  - {piece}")
        lines.append("Panel Controls: toggle_panel('quotes') / toggle_panel('ladders') etc.")
        return "\n".join(lines)

    def _render_simulations(self, context: DashboardContext) -> str:
        simulations = context.simulations
        if not simulations:
            return "No simulation results available."
        lines = ["Event        Home   Away   H Win  A Win   Tie   Margin  Total"]
        for result in simulations:
            lines.append(
                f"{result.event_id:<12}{result.home_team:<6}{result.away_team:<6}"
                f"{result.home_win_probability:>6.2%}{result.away_win_probability:>6.2%}"
                f"{result.tie_probability():>6.2%}{result.expected_margin:>8.2f}{result.expected_total:>7.2f}"
            )
        return "\n".join(lines)

    def _render_quotes(self, context: DashboardContext) -> str:
        odds = sorted(context.odds, key=lambda q: q.observed_at, reverse=True)
        if not odds:
            return "No stored odds quotes."
        lines = [
            "Book     Market Group        Market        Scope  Selection             Side   Line   Odds   Seen",
        ]
        for quote in odds[:20]:
            line_display = f"{quote.line:.1f}" if quote.line is not None else "-"
            side_display = quote.side or "-"
            seen = quote.observed_at.strftime("%H:%M:%S")
            lines.append(
                f"{quote.sportsbook:<8}{quote.book_market_group:<18}{quote.market:<13}"
                f"{quote.scope:<6}{quote.team_or_player:<20}{side_display:<6}{line_display:>6}"
                f"{quote.american_odds:>7}{seen:>7}"
            )
            return "\n".join(lines)

    def _render_opportunities(self, context: DashboardContext) -> str:
        opportunities = context.opportunities
        if not opportunities:
            return "No actionable opportunities detected."
        lines = [
            "Event      Market       Scope Selection            Odds   Model   Push  Implied    EV    Kelly",
        ]
        for opp in opportunities[:20]:
            side = opp.side or "-"
            selection = f"{opp.team_or_player} {side}".strip()
            line_display = ""
            if opp.line is not None:
                line_display = f" {opp.line:.1f}"
            lines.append(
                f"{opp.event_id:<10}{opp.market:<12}{opp.scope:<6}{selection:<20}{opp.american_odds:>6}"
                f"{opp.model_probability:>8.2%}{opp.push_probability:>7.2%}{opp.implied_probability:>8.2%}"
                f"{opp.expected_value:>7.2%}{opp.kelly_fraction:>8.2%}{line_display}"
            )
        return "\n".join(lines)

    def _render_risk(self, context: DashboardContext) -> str:
        summary = context.risk_summary
        if summary is None:
            return "Risk analytics unavailable."
        lines = [
            f"Configured bankroll: {summary.bankroll:,.2f} units",
            f"Opportunity Kelly fraction: {summary.opportunity_fraction:.2f}",
            f"Portfolio Kelly fraction: {summary.portfolio_fraction:.2f}",
        ]
        positions = summary.positions
        if not positions:
            lines.append("No active positions.")
        else:
            total_stake = sum(position.stake for position in positions)
            lines.append(
                f"Open positions: {len(positions)} | Total stake: {total_stake:,.2f}"
            )
            for position in positions[:5]:
                opp = position.opportunity
                lines.append(
                    f"  {opp.event_id} {opp.market} {opp.team_or_player}"
                    f" @ {opp.american_odds:+d} stake={position.stake:.2f}"
                )
            if len(positions) > 5:
                lines.append(f"  … {len(positions) - 5} additional positions")
        exposure = summary.exposure_by_event
        if exposure:
            lines.append("Event exposure:")
            for key, stake in sorted(exposure.items()):
                fraction = (stake / summary.bankroll) if summary.bankroll else 0.0
                lines.append(
                    f"  {key[0]} {key[1]} -> {stake:,.2f} ({fraction:.2%} of bankroll)"
                )
        correlation = summary.correlation_exposure
        if correlation:
            lines.append("Correlation exposure:")
            for group, stake in sorted(correlation.items()):
                fraction = (stake / summary.bankroll) if summary.bankroll else 0.0
                lines.append(f"  {group}: {stake:,.2f} ({fraction:.2%})")
        if summary.simulation:
            metrics = summary.simulation.summary()
            lines.append("Simulation drawdowns:")
            lines.append(f"  Trials: {int(metrics['trials'])}")
            lines.append(f"  Mean terminal: {metrics['mean_terminal']:.2f}")
            lines.append(f"  Median terminal: {metrics['median_terminal']:.2f}")
            lines.append(f"  Worst terminal: {metrics['worst_terminal']:.2f}")
            lines.append(
                f"  Average drawdown: {metrics['average_drawdown']:.2%}"
            )
            lines.append(f"  Worst drawdown: {metrics['worst_drawdown']:.2%}")
            lines.append(
                f"  5th percentile drawdown: {metrics['p05_drawdown']:.2%}"
            )
            lines.append(
                f"  95th percentile drawdown: {metrics['p95_drawdown']:.2%}"
            )
        return "\n".join(lines)

    def _render_ladders(self, context: DashboardContext) -> str:
        ladders = _build_ladder_matrix(context.odds)
        if not ladders:
            return ""
        sections: list[str] = []
        for (event_id, market, selection), ladder in ladders.items():
            scopes = sorted(ladder.keys())
            sections.append(f"{event_id} · {market} · {selection}")
            header = "Line  " + "".join(f"{scope:>9}" for scope in scopes)
            sections.append(header)
            lines = sorted({line for entries in ladder.values() for line in entries})
            for line_value in lines:
                row = f"{line_value:>5.1f} "
                for scope in scopes:
                    odds_value = ladder[scope].get(line_value)
                    if odds_value is None:
                        row += " " * 9
                    else:
                        row += f"{odds_value:>9}"
                sections.append(row.rstrip())
            sections.append("")
        return "\n".join(section for section in sections).strip()

    def _render_search(self, context: DashboardContext) -> str:
        query = context.search.query
        if not query:
            return "No active search query."
        lines = [f"Query: {query!r}"]
        if not any(context.search_results.values()):
            lines.append("No matches found across active panels.")
            return "\n".join(lines)
        for bucket, matches in context.search_results.items():
            if not matches:
                continue
            lines.append("")
            lines.append(bucket.title())
            lines.append("~" * len(bucket))
            for match in matches[:10]:
                if isinstance(match, IngestedOdds):
                    seen = match.observed_at.strftime("%H:%M:%S")
                    lines.append(
                        f"{match.event_id} · {match.market} · {match.team_or_player}"
                        f" @ {match.sportsbook} ({match.scope}, {match.american_odds:+d}) [{seen}]"
                    )
                elif isinstance(match, Opportunity):
                    lines.append(
                        f"{match.event_id} · {match.market} · {match.team_or_player}"
                        f" {match.american_odds:+d} EV={match.expected_value:0.2%}"
                    )
                else:
                    lines.append(
                        f"{match.event_id} · {match.home_team} vs {match.away_team}"
                        f" H={match.home_win_probability:0.1%} A={match.away_win_probability:0.1%}"
                    )
        return "\n".join(lines)

    def _apply_search(
        self,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
    ) -> dict[str, Sequence[object]]:
        if not self.search.query:
            return {"quotes": [], "opportunities": [], "simulations": []}

        def _match_quote(quote: IngestedOdds) -> bool:
            haystack = " ".join(
                [
                    quote.event_id,
                    quote.sportsbook,
                    quote.book_market_group,
                    quote.market,
                    quote.scope,
                    quote.team_or_player,
                    quote.side or "",
                ]
            )
            return self.search.match(haystack)

        def _match_opportunity(opp: Opportunity) -> bool:
            haystack = " ".join(
                [
                    opp.event_id,
                    opp.sportsbook,
                    opp.market,
                    opp.scope,
                    opp.team_or_player,
                    opp.side or "",
                ]
            )
            return self.search.match(haystack)

        def _match_simulation(result: SimulationResult) -> bool:
            haystack = " ".join([result.event_id, result.home_team, result.away_team])
            return self.search.match(haystack)

        matches: dict[str, Sequence[object]] = {"quotes": [], "opportunities": [], "simulations": []}
        if "quotes" in self.search.targets:
            matches["quotes"] = [quote for quote in odds if _match_quote(quote)]
        if "opportunities" in self.search.targets:
            matches["opportunities"] = [opp for opp in opportunities if _match_opportunity(opp)]
        if "simulations" in self.search.targets:
            matches["simulations"] = [sim for sim in simulations if _match_simulation(sim)]
        return matches


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _freeze_optional(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(values)


def _normalize_scope(scope: str) -> str:
    return scope.strip().lower().replace(" ", "")


def _is_quarter_scope(scope: str) -> bool:
    normalized = scope.replace("quarter", "q")
    return normalized in {
        "q1",
        "1q",
        "firstq",
        "q2",
        "2q",
        "secondq",
        "q3",
        "3q",
        "thirdq",
        "q4",
        "4q",
        "fourthq",
    }


def _is_half_scope(scope: str) -> bool:
    normalized = scope.replace("half", "h")
    return normalized in {
        "h1",
        "1h",
        "firsth",
        "h2",
        "2h",
        "secondh",
        "fh",
        "sh",
    }


def _describe_dimension(name: str, values: frozenset[str] | None) -> str:
    if not values:
        return f"{name}: All"
    return f"{name}: {', '.join(sorted(values))}"


def _build_ladder_matrix(
    odds: Sequence[IngestedOdds],
) -> dict[tuple[str, str, str], dict[str, dict[float, int]]]:
    ladder: dict[tuple[str, str, str], dict[str, dict[float, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for quote in odds:
        if quote.line is None:
            continue
        key = (quote.event_id, quote.market, quote.team_or_player)
        scope = _normalize_scope(quote.scope)
        ladder[key][scope][round(float(quote.line), 1)] = quote.american_odds
    return {k: dict(v) for k, v in ladder.items()}


def _parse_filter_tokens(tokens: Sequence[str]) -> dict[str, object]:
    updates: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid filter token: {token}")
        key, value = token.split("=", 1)
        key = key.strip()
        values = [item.strip() for item in value.split(",") if item.strip()]
        updates[key] = None if not values else values
    return updates


class TerminalDashboardSession:
    """Small command interpreter for interactive terminal dashboards."""

    def __init__(self, dashboard: Dashboard | None = None) -> None:
        self.dashboard = dashboard or Dashboard()

    @property
    def panels(self) -> Sequence[str]:
        return tuple(self.dashboard._panel_order)

    def handle(
        self,
        command: str,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
    ) -> str:
        parts = shlex.split(command)
        if not parts:
            return self.dashboard.render(odds, simulations, opportunities)
        action = parts[0].lower()
        args = parts[1:]
        if action in {"help", "?"}:
            return self._help()
        if action == "show":
            return self.dashboard.render(odds, simulations, opportunities)
        if action == "reset":
            self.dashboard.reset_filters()
            self.dashboard.reset_search()
            return "Filters and search reset."
        if action == "toggle":
            return self._toggle(args)
        if action == "filter":
            return self._filter(args)
        if action == "search":
            return self._search(args)
        if action == "clear" and args and args[0] == "search":
            self.dashboard.reset_search()
            return "Search cleared."
        if action == "panels":
            return "Available panels: " + ", ".join(self.dashboard._panels)
        if action == "order":
            return self._order(args)
        if action == "panel":
            return self._panel(args)
        return f"Unknown command: {action}. Type 'help' for available commands."

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _help(self) -> str:
        return (
            "Commands:\n"
            "  show                            Render the dashboard\n"
            "  filter key=value[...]            Apply filters (comma separated values)\n"
            "  toggle quarters|halves|panel KEY Toggle quarters/halves or a panel\n"
            "  search <query> [--case] [--in targets]   Apply text search\n"
            "  clear search                   Remove search query\n"
            "  order panel1,panel2,...        Reorder panels\n"
            "  panels                          List available panels\n"
            "  reset                           Reset filters and search"
        )

    def _toggle(self, args: Sequence[str]) -> str:
        if not args:
            return "Toggle requires an argument (quarters, halves, or panel <key>)."
        target = args[0].lower()
        if target == "quarters":
            state = self.dashboard.toggle_quarters()
            return f"Quarter markets {'enabled' if state else 'disabled'}."
        if target == "halves":
            state = self.dashboard.toggle_halves()
            return f"Half markets {'enabled' if state else 'disabled'}."
        if target == "panel" and len(args) > 1:
            visible = self.dashboard.toggle_panel(args[1])
            return f"Panel '{args[1]}' {'expanded' if visible else 'collapsed'}."
        return "Unsupported toggle argument."

    def _filter(self, args: Sequence[str]) -> str:
        if not args:
            return "Usage: filter sportsbooks=DK,FanDuel markets=moneyline"
        try:
            updates = _parse_filter_tokens(args)
        except ValueError as exc:
            return str(exc)
        try:
            self.dashboard.set_filters(**updates)
        except TypeError as exc:
            return f"Error applying filters: {exc}"
        return "Filters updated."

    def _search(self, args: Sequence[str]) -> str:
        if not args:
            return "Usage: search <query> [--case] [--in quotes,opportunities]"
        query: list[str] = []
        case_sensitive = None
        targets: Iterable[str] | None = None
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--case":
                case_sensitive = True
            elif token == "--nocase":
                case_sensitive = False
            elif token == "--in":
                index += 1
                if index >= len(args):
                    return "Missing argument for --in"
                targets = [part.strip() for part in args[index].split(",") if part.strip()]
            else:
                query.append(token)
            index += 1
        final_query = " ".join(query)
        self.dashboard.set_search(final_query, case_sensitive=case_sensitive, targets=targets)
        return f"Search set to {final_query!r}."

    def _order(self, args: Sequence[str]) -> str:
        if not args:
            return "Usage: order controls,quotes,..."
        order = [part.strip() for part in args[0].split(",") if part.strip()]
        try:
            self.dashboard.reorder_panels(order)
        except KeyError as exc:
            return str(exc)
        return "Panel order updated."

    def _panel(self, args: Sequence[str]) -> str:
        if len(args) < 2 or args[0] != "info":
            return "Usage: panel info <key>"
        key = args[1]
        panel = self.dashboard._panels.get(key)
        if not panel:
            return f"Unknown panel '{key}'."
        return f"Panel '{panel.title}' is {'collapsed' if panel.collapsed else 'expanded'}."

