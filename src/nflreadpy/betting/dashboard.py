"""ASCII dashboard inspired by Bloomberg-style terminal layouts.

This module implements an interactive console dashboard with filter controls,
panel toggles, and ladder matrix summaries.  The same filtering logic is reused
by the Streamlit web dashboard, allowing the two interfaces to stay in sync.
"""

from __future__ import annotations

import datetime as dt
import shlex
from dataclasses import dataclass
from typing import Iterable, Sequence

from .analytics import Opportunity
from .dashboard_core import (
    DEFAULT_SEARCH_TARGETS,
    DashboardContext,
    DashboardFilters,
    LadderCell,
    DashboardPanelState,
    DashboardPanelView,
    DashboardSearchState,
    DashboardSnapshot,
    RiskSummary,
    build_ladder_matrix,
    is_half_scope,
    is_quarter_scope,
    normalize_scope,
    parse_filter_tokens,
)
from .ingestion import IngestedOdds
from .models import SimulationResult
from .scrapers.base import american_to_decimal


@dataclass(frozen=True, slots=True)
class DashboardHotkey:
    """Describe a hotkey binding shared between terminal surfaces."""

    key: str
    command: str
    description: str


# Backwards compatibility aliases for historical private imports
_build_ladder_matrix = build_ladder_matrix
_parse_filter_tokens = parse_filter_tokens
_normalize_scope = normalize_scope
_is_quarter_scope = is_quarter_scope
_is_half_scope = is_half_scope


_SCOPE_PRESETS: dict[str, dict[str, object]] = {
    "all": {
        "include_quarters": True,
        "include_halves": True,
        "scopes": None,
    },
    "game": {
        "include_quarters": False,
        "include_halves": False,
        "scopes": ("game",),
    },
    "main": {
        "include_quarters": False,
        "include_halves": True,
        "scopes": ("game", "1st half", "2nd half"),
    },
}


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
        self._hotkeys: tuple[DashboardHotkey, ...] | None = None

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

    def scope_presets(self) -> tuple[str, ...]:
        """Return the available scope presets recognised by the dashboard."""

        return tuple(sorted(_SCOPE_PRESETS))

    def apply_scope_preset(self, preset: str) -> DashboardFilters:
        """Apply a named scope preset and return the updated filters."""

        key = preset.lower()
        if key not in _SCOPE_PRESETS:
            valid = ", ".join(self.scope_presets())
            raise KeyError(f"Unknown scope preset '{preset}'. Available presets: {valid}")
        updates = dict(_SCOPE_PRESETS[key])
        scopes = updates.pop("scopes")
        self.filters = self.filters.update(scopes=scopes, **updates)
        return self.filters

    # ------------------------------------------------------------------
    # Hotkey bindings
    # ------------------------------------------------------------------
    def hotkey_bindings(self) -> tuple[DashboardHotkey, ...]:
        """Return the default hotkey bindings used by terminal interfaces."""

        if self._hotkeys is None:
            self._hotkeys = (
                DashboardHotkey("s", "show", "Render the dashboard using current filters."),
                DashboardHotkey("f", "filter", "Apply filter tokens (append key=value arguments)."),
                DashboardHotkey("/", "search", "Apply a search query across panels."),
                DashboardHotkey("n", "clear search", "Clear the active search query."),
                DashboardHotkey("c", "reset", "Reset filters and search state."),
                DashboardHotkey("q", "toggle quarters", "Toggle quarter markets."),
                DashboardHotkey("h", "toggle halves", "Toggle half markets."),
                DashboardHotkey("g", "scope game", "Apply the game-only scope preset."),
                DashboardHotkey("m", "scope main", "Apply the main scope preset (game + halves)."),
                DashboardHotkey("a", "scope all", "Show all scopes, including quarters."),
                DashboardHotkey("l", "toggle panel ladders", "Collapse or expand the ladder panel."),
                DashboardHotkey("o", "toggle panel opportunities", "Collapse or expand the opportunities panel."),
            )
        return self._hotkeys

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
        lines = ["Book     Market Group        Market        Scope  Selection             Side   Line   Odds   Seen"]
        for quote in odds[:20]:
            line_display = f"{quote.line:.1f}" if quote.line is not None else "-"
            side_display = quote.side or "-"
            seen = quote.observed_at.strftime("%H:%M:%S")
            lines.append(
                f"{quote.sportsbook:<8}{quote.book_market_group:<18}{quote.market:<13}"
                f"{quote.scope:<6}{quote.team_or_player:<20}{side_display:<6}{line_display:>6}"
                f"{quote.american_odds:>7}{seen:>7}"
            )
        return "\n".join(lines)  # <- outdented

    def _render_opportunities(self, context: DashboardContext) -> str:
        opportunities = context.opportunities
        if not opportunities:
            return "No actionable opportunities detected."
        lines = [
            "Event      Market       Scope Selection            US    Dec    Frac   Model   Push  Implied    EV    Kelly",
        ]
        for opp in opportunities[:20]:
            side = opp.side or "-"
            selection = f"{opp.team_or_player} {side}".strip()
            line_display = ""
            if opp.line is not None:
                line_display = f" {opp.line:.1f}"
            decimal = opp.decimal_odds()
            frac_num, frac_den = opp.fractional_odds()
            frac_display = f"{frac_num}/{frac_den}"
            lines.append(
                f"{opp.event_id:<10}{opp.market:<12}{opp.scope:<6}{selection:<20}{opp.american_odds:>6}"
                f"{decimal:>7.2f}{frac_display:>8}{opp.model_probability:>8.2%}"
                f"{opp.push_probability:>7.2%}{opp.implied_probability:>8.2%}"
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
        if summary.correlation_limits:
            lines.append("Correlation limits:")
            for group, fraction in sorted(summary.correlation_limits.items()):
                lines.append(f"  {group}: {fraction:.2%} of bankroll")
        metrics = summary.bankroll_summary
        if metrics is None and summary.simulation:
            metrics = summary.simulation.summary()
        if metrics:
            lines.append("Simulation drawdowns:")
            lines.append(f"  Trials: {int(metrics['trials'])}")
            lines.append(f"  Mean terminal: {metrics['mean_terminal']:.2f}")
            lines.append(f"  Median terminal: {metrics['median_terminal']:.2f}")
            lines.append(f"  Worst terminal: {metrics['worst_terminal']:.2f}")
            lines.append(f"  Average drawdown: {metrics['average_drawdown']:.2%}")
            lines.append(f"  Worst drawdown: {metrics['worst_drawdown']:.2%}")
            lines.append(f"  5th percentile drawdown: {metrics['p05_drawdown']:.2%}")
            lines.append(f"  95th percentile drawdown: {metrics['p95_drawdown']:.2%}")
        return "\n".join(lines)

    def _render_ladders(self, context: DashboardContext) -> str:
        ladders = _build_ladder_matrix(context.odds)
        if not ladders:
            return ""
        sections: list[str] = []
        for (event_id, market, selection), ladder in ladders.items():
            scopes = sorted(ladder.keys())
            sections.append(f"{event_id} · {market} · {selection}")
            cell_width = 14
            header = "Line  " + "".join(f"{scope:>{cell_width}}" for scope in scopes) + "  Best"
            sections.append(header)
            lines = sorted({line for entries in ladder.values() for line in entries})
            for line_value in lines:
                row = f"{line_value:>5.1f} "
                best_scope = self._best_scope_for_line(ladder, line_value)
                for scope in scopes:
                    cell = ladder[scope].get(line_value)
                    if cell is None:
                        row += " " * cell_width
                        continue
                    marker = "*" if scope == best_scope else " "
                    display = cell.summary()
                    row += f"{marker}{display:>{cell_width - 1}}"
                row += f"  {best_scope or '-'}"
                sections.append(row.rstrip())
            sections.append("")
        return "\n".join(section for section in sections).strip()

    @staticmethod
    def _best_scope_for_line(
        ladder: dict[str, dict[float, LadderCell]], line_value: float
    ) -> str | None:
        best_scope: str | None = None
        best_price: float | None = None
        for scope, entries in ladder.items():
            cell = entries.get(line_value)
            if cell is None:
                continue
            decimal = american_to_decimal(cell.american_odds)
            if best_price is None or decimal > best_price:
                best_price = decimal
                best_scope = scope
        return best_scope

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


class TerminalDashboardSession:
    """Small command interpreter for interactive terminal dashboards."""

    def __init__(self, dashboard: Dashboard | None = None) -> None:
        self.dashboard = dashboard or Dashboard()
        self._hotkey_map: dict[str, DashboardHotkey] = {
            binding.key: binding for binding in self.dashboard.hotkey_bindings()
        }

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
        if action == "scope":
            return self._scope(args)
        if action == "hotkeys":
            return self._hotkey_help()
        if action == "hotkey":
            return self._hotkey(args, odds, simulations, opportunities)
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
            "  scope <preset>                  Apply scope preset (all, game, main)\n"
            "  order panel1,panel2,...        Reorder panels\n"
            "  panels                          List available panels\n"
            "  hotkeys                        Show available hotkeys\n"
            "  hotkey <key> [args...]         Dispatch a hotkey binding\n"
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

    def _scope(self, args: Sequence[str]) -> str:
        if not args:
            presets = ", ".join(self.dashboard.scope_presets())
            return f"Usage: scope <preset>. Available presets: {presets}"
        preset = args[0]
        try:
            self.dashboard.apply_scope_preset(preset)
        except KeyError as exc:
            return str(exc)
        return f"Scope preset '{preset}' applied."

    def _hotkey(
        self,
        args: Sequence[str],
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
    ) -> str:
        if not args:
            return self._hotkey_help()
        key = args[0]
        binding = self._hotkey_map.get(key)
        if binding is None:
            return f"Unknown hotkey '{key}'. Type 'hotkeys' to list bindings."
        command = binding.command
        if len(args) > 1:
            command = " ".join((command, *args[1:]))
        return self.handle(command, odds, simulations, opportunities)

    def _hotkey_help(self) -> str:
        bindings = sorted(self._hotkey_map.values(), key=lambda item: item.key)
        lines = ["Hotkey bindings:"]
        for binding in bindings:
            lines.append(f"  {binding.key} -> {binding.command} — {binding.description}")
        lines.append("Use 'hotkey <key>' (optionally followed by arguments) to invoke a binding.")
        return "\n".join(lines)

