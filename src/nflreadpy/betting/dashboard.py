"""ASCII dashboard inspired by Bloomberg terminal layouts."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

from .analytics import Opportunity
from .ingestion import IngestedOdds
from .models import SimulationResult


class Dashboard:
    """Render odds, model outputs, and opportunities in tabular form."""

    def render(
        self,
        odds: Sequence[IngestedOdds],
        simulations: Sequence[SimulationResult],
        opportunities: Sequence[Opportunity],
    ) -> str:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        header = f"NFL Terminal — {now}\n"
        header += "=" * 92 + "\n"
        body = [header]
        body.append(self._render_simulations(simulations))
        body.append(self._render_odds_snapshot(odds))
        body.append(self._render_opportunities(opportunities))
        return "\n".join(part for part in body if part)

    def _render_simulations(self, simulations: Sequence[SimulationResult]) -> str:
        if not simulations:
            return "No simulation results available."
        lines = [
            "Simulations",
            "------------",
            "Event        Home   Away   H Win  A Win   Tie   Margin  Total",
        ]
        for result in simulations:
            lines.append(
                f"{result.event_id:<12}{result.home_team:<6}{result.away_team:<6}"
                f"{result.home_win_probability:>6.2%}{result.away_win_probability:>6.2%}"
                f"{result.tie_probability():>6.2%}{result.expected_margin:>8.2f}{result.expected_total:>7.2f}"
            )
        return "\n".join(lines)

    def _render_odds_snapshot(self, odds: Sequence[IngestedOdds]) -> str:
        if not odds:
            return "\nNo stored odds quotes."
        lines = [
            "",
            "Latest Quotes",
            "-------------",
            "Book     Market Group        Market        Scope  Selection             Side   Line   Odds",
        ]
        for quote in odds[:12]:
            line_display = f"{quote.line:.1f}" if quote.line is not None else "-"
            side_display = quote.side or "-"
            lines.append(
                f"{quote.sportsbook:<8}{quote.book_market_group:<18}{quote.market:<13}"
                f"{quote.scope:<6}{quote.team_or_player:<20}{side_display:<6}{line_display:>6}{quote.american_odds:>7}"
            )
        return "\n".join(lines)

    def _render_opportunities(self, opportunities: Sequence[Opportunity]) -> str:
        if not opportunities:
            return "\nNo actionable opportunities detected."
        lines = [
            "",
            "Opportunities",
            "--------------",
            "Event      Market       Scope Selection            Odds   Model   Push  Implied    EV    Kelly",
        ]
        for opp in opportunities:
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

