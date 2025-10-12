"""Curses-powered terminal user interface for the betting dashboard."""

from __future__ import annotations

import curses
import time
from typing import Protocol, Sequence

from .analytics import Opportunity
from .dashboard import Dashboard, DashboardSnapshot, RiskSummary, _parse_filter_tokens
from .ingestion import IngestedOdds
from .models import SimulationResult


class DashboardFeed(Protocol):
    """Minimal protocol satisfied by dashboard data providers."""

    def live_markets(self) -> Sequence[IngestedOdds]:
        ...

    def simulations(self) -> Sequence[SimulationResult]:  # pragma: no cover - optional
        ...

    def opportunities(self) -> Sequence[Opportunity]:  # pragma: no cover - optional
        ...

    def risk(self) -> RiskSummary | None:  # pragma: no cover - optional
        ...

    def line_history(self) -> Sequence[IngestedOdds]:  # pragma: no cover - optional
        ...


class DashboardKeyboardController:
    """State container for keyboard-driven dashboard sessions."""

    def __init__(self, dashboard: Dashboard | None = None) -> None:
        self.dashboard = dashboard or Dashboard()
        self.selected_panel = 0
        self.status = "Press ? for help."
        self.last_snapshot: DashboardSnapshot | None = None

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    @property
    def panel_keys(self) -> tuple[str, ...]:
        return tuple(self.dashboard._panel_order)

    @property
    def current_panel_key(self) -> str:
        return self.panel_keys[self.selected_panel]

    def focus_next_panel(self) -> str:
        self.selected_panel = (self.selected_panel + 1) % len(self.panel_keys)
        return self.current_panel_key

    def focus_previous_panel(self) -> str:
        self.selected_panel = (self.selected_panel - 1) % len(self.panel_keys)
        return self.current_panel_key

    def toggle_current_panel(self) -> bool:
        key = self.current_panel_key
        visible = self.dashboard.toggle_panel(key)
        self.status = f"Panel '{key}' {'expanded' if visible else 'collapsed'}."
        return visible

    def toggle_quarters(self) -> bool:
        state = self.dashboard.toggle_quarters()
        self.status = f"Quarter markets {'enabled' if state else 'disabled'}."
        return state

    def toggle_halves(self) -> bool:
        state = self.dashboard.toggle_halves()
        self.status = f"Half markets {'enabled' if state else 'disabled'}."
        return state

    def apply_filter_expression(self, expression: str) -> str:
        expression = expression.strip()
        if not expression:
            self.dashboard.reset_filters()
            self.status = "Filters reset."
            return self.status
        tokens = [token for token in expression.split() if token]
        try:
            updates = _parse_filter_tokens(tokens)
        except ValueError as exc:  # pragma: no cover - defensive guard
            self.status = str(exc)
            return self.status
        self.dashboard.set_filters(**updates)
        self.status = "Filters updated."
        return self.status

    def reset_filters(self) -> None:
        self.dashboard.reset_filters()
        self.status = "Filters reset."

    def apply_search(self, query: str | None) -> None:
        query = query.strip() if query else None
        self.dashboard.set_search(query)
        if query:
            self.status = f"Search set to {query!r}."
        else:
            self.status = "Search cleared."

    def clear_search(self) -> None:
        self.dashboard.reset_search()
        self.status = "Search cleared."

    def apply_scope_preset(self, preset: str) -> None:
        try:
            self.dashboard.apply_scope_preset(preset)
        except KeyError as exc:  # pragma: no cover - defensive guard
            self.status = str(exc)
            return
        self.status = f"Scope preset '{preset}' applied."

    def refresh(
        self,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
        movement_history: Sequence[IngestedOdds] | None = None,
        *,
        risk_summary: RiskSummary | None = None,
        movement_depth: int | None = None,
        movement_threshold: int | None = None,
        stale_after: dt.timedelta | None = None,
    ) -> DashboardSnapshot:
        snapshot = self.dashboard.snapshot(
            odds,
            simulations,
            opportunities,
            risk_summary=risk_summary,
            movement_history=movement_history,
            movement_depth=movement_depth,
            movement_threshold=movement_threshold,
            stale_after=stale_after,
        )
        self.last_snapshot = snapshot
        return snapshot


# ----------------------------------------------------------------------
# Curses application
# ----------------------------------------------------------------------


def run_curses_dashboard(
    feed: DashboardFeed,
    *,
    refresh_seconds: float = 5.0,
) -> None:
    """Run an interactive curses dashboard backed by ``feed``."""

    app = _CursesDashboardApp(feed, max(refresh_seconds, 0.5))
    curses.wrapper(app.run)


class _CursesDashboardApp:
    def __init__(self, feed: DashboardFeed, refresh_seconds: float) -> None:
        self.feed = feed
        self.refresh_seconds = refresh_seconds
        self.controller = DashboardKeyboardController()
        self._needs_refresh = True

    def run(self, screen: "curses._CursesWindow") -> None:  # type: ignore[name-defined]
        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(200)
        snapshot = self._refresh_controller()
        self._draw(screen, snapshot)
        last_refresh = time.monotonic()
        while True:
            now = time.monotonic()
            if self._needs_refresh or (now - last_refresh) >= self.refresh_seconds:
                snapshot = self._refresh_controller()
                self._draw(screen, snapshot)
                self._needs_refresh = False
                last_refresh = now
            key = screen.getch()
            if key == -1:
                continue
            if self._handle_key(screen, key):
                break

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _draw(self, screen: "curses._CursesWindow", snapshot: DashboardSnapshot) -> None:  # type: ignore[name-defined]
        screen.erase()
        max_y, max_x = screen.getmaxyx()
        row = 0
        for index, line in enumerate(snapshot.header):
            attr = curses.A_BOLD if index == 0 else curses.A_DIM
            screen.addnstr(row, 0, line[: max_x - 1], max_x - 1, attr)
            row += 1
        row += 1
        for idx, view in enumerate(snapshot.panels):
            if row >= max_y - 2:
                break
            highlight = curses.A_REVERSE if idx == self.controller.selected_panel else curses.A_NORMAL
            title = view.state.title + (" (collapsed)" if view.state.collapsed else "")
            screen.addnstr(row, 0, title[: max_x - 1], max_x - 1, curses.A_BOLD | highlight)
            row += 1
            separator = "-" * min(len(view.state.title), max_x - 1)
            screen.addnstr(row, 0, separator, max_x - 1, highlight)
            row += 1
            if not view.state.collapsed:
                body = view.body or ("No data available.",)
                for line in body:
                    if row >= max_y - 2:
                        break
                    screen.addnstr(row, 0, line[: max_x - 1], max_x - 1)
                    row += 1
            row += 1
        help_line = (
            "Tab⇆ navigate · Space toggle · f filter · / search · Q quarters · H halves · "
            "g game scope · m main scope · a all scopes · c reset filters · n clear search · r refresh · q quit"
        )
        screen.addnstr(max_y - 2, 0, help_line[: max_x - 1], max_x - 1, curses.A_DIM)
        status = self.controller.status
        screen.addnstr(max_y - 1, 0, status[: max_x - 1].ljust(max_x - 1), max_x - 1, curses.A_REVERSE)
        screen.refresh()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _handle_key(self, screen: "curses._CursesWindow", key: int) -> bool:  # type: ignore[name-defined]
        if key in (ord("q"), 27):
            self.controller.status = "Exiting dashboard."
            self._draw(screen, self.controller.last_snapshot)  # type: ignore[arg-type]
            return True
        if key in (curses.KEY_TAB, 9):
            focused = self.controller.focus_next_panel()
            self.controller.status = f"Focused panel: {focused}"
            self._draw(screen, self.controller.last_snapshot)  # type: ignore[arg-type]
            return False
        if key in (curses.KEY_BTAB, 353):
            focused = self.controller.focus_previous_panel()
            self.controller.status = f"Focused panel: {focused}"
            self._draw(screen, self.controller.last_snapshot)  # type: ignore[arg-type]
            return False
        if key in (ord(" "), curses.KEY_ENTER, 10, 13):
            self.controller.toggle_current_panel()
            self._draw(screen, self.controller.last_snapshot)  # type: ignore[arg-type]
            return False
        if key in (ord("Q"),):
            self.controller.toggle_quarters()
            self._needs_refresh = True
            return False
        if key in (ord("H"),):
            self.controller.toggle_halves()
            self._needs_refresh = True
            return False
        if key in (ord("g"),):
            self.controller.apply_scope_preset("game")
            self._needs_refresh = True
            return False
        if key in (ord("m"),):
            self.controller.apply_scope_preset("main")
            self._needs_refresh = True
            return False
        if key in (ord("a"),):
            self.controller.apply_scope_preset("all")
            self._needs_refresh = True
            return False
        if key in (ord("f"),):
            expression = self._prompt(screen, "Filter expression: ")
            if expression is not None:
                self.controller.apply_filter_expression(expression)
                self._needs_refresh = True
            return False
        if key in (ord("/"),):
            query = self._prompt(screen, "Search query: ")
            self.controller.apply_search(query)
            self._needs_refresh = True
            return False
        if key in (ord("n"),):
            self.controller.clear_search()
            self._needs_refresh = True
            return False
        if key in (ord("c"),):
            self.controller.reset_filters()
            self._needs_refresh = True
            return False
        if key in (ord("r"),):
            self._needs_refresh = True
            return False
        if key in (ord("?"),):
            self.controller.status = (
                "Keys: Tab navigate | Shift+Tab back | Space toggle | f filter | / search | "
                "Q quarters | H halves | g game scope | m main scope | a all scopes | c reset filters | n clear search | r refresh | q quit"
            )
            self._draw(screen, self.controller.last_snapshot)  # type: ignore[arg-type]
            return False
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _prompt(self, screen: "curses._CursesWindow", prompt: str) -> str | None:  # type: ignore[name-defined]
        max_y, max_x = screen.getmaxyx()
        row = max(max_y - 3, 0)
        screen.move(row, 0)
        screen.clrtoeol()
        screen.addnstr(row, 0, prompt[: max_x - 1], max_x - 1, curses.A_BOLD)
        screen.refresh()
        curses.echo()
        try:
            column = max(min(len(prompt), max_x - 2), 0)
            response = screen.getstr(row, column).decode().strip()
        except Exception:  # pragma: no cover - defensive
            response = ""
        finally:
            curses.noecho()
        screen.move(row, 0)
        screen.clrtoeol()
        return response or None

    def _load_data(
        self,
    ) -> tuple[
        Sequence[IngestedOdds],
        Sequence[SimulationResult],
        Sequence[Opportunity],
        RiskSummary | None,
        Sequence[IngestedOdds],
    ]:
        odds = list(self.feed.live_markets())
        simulations_fetcher = getattr(self.feed, "simulations", lambda: [])
        opportunities_fetcher = getattr(self.feed, "opportunities", lambda: [])
        risk_fetcher = getattr(self.feed, "risk", lambda: None)
        history_fetcher = getattr(self.feed, "line_history", lambda: [])
        simulations = list(simulations_fetcher())
        opportunities = list(opportunities_fetcher())
        risk_summary = risk_fetcher()
        movement_history = list(history_fetcher())
        return odds, simulations, opportunities, risk_summary, movement_history

    def _refresh_controller(self) -> DashboardSnapshot:
        odds, simulations, opportunities, risk_summary, movement_history = self._load_data()
        return self.controller.refresh(
            odds,
            simulations,
            opportunities,
            movement_history,
            risk_summary=risk_summary,
        )
