from __future__ import annotations

import datetime as dt

import pytest

import nflreadpy.betting.analytics as analytics_module
from nflreadpy.betting.dashboard import (
    Dashboard,
    DashboardSnapshot,
    RiskSummary,
    TerminalDashboardSession,
)
from nflreadpy.betting.dashboard_core import build_ladder_matrix
from nflreadpy.betting.dashboard_tui import DashboardKeyboardController
from nflreadpy.betting import dashboard as dashboard_module
from nflreadpy.betting.analytics import Opportunity, PortfolioPosition
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.models import SimulationResult


@pytest.fixture
def sample_quotes() -> list[IngestedOdds]:
    observed = dt.datetime(2024, 1, 21, 12, 0, tzinfo=dt.timezone.utc)
    return [
        IngestedOdds(
            event_id="KC@BUF",
            sportsbook="FanDuel",
            book_market_group="moneyline",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="Kansas City Chiefs",
            side=None,
            line=None,
            american_odds=-120,
            observed_at=observed,
            extra={},
        ),
        IngestedOdds(
            event_id="KC@BUF",
            sportsbook="DraftKings",
            book_market_group="spreads",
            market="spread",
            scope="1st half",
            entity_type="team",
            team_or_player="Buffalo Bills",
            side="home",
            line=-2.5,
            american_odds=-105,
            observed_at=observed,
            extra={},
        ),
    ]


@pytest.fixture
def sample_opportunities() -> list[Opportunity]:
    return [
        Opportunity(
            event_id="KC@BUF",
            sportsbook="FanDuel",
            book_market_group="moneyline",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="Kansas City Chiefs",
            side=None,
            line=None,
            american_odds=-120,
            model_probability=0.58,
            push_probability=0.0,
            implied_probability=0.545,
            expected_value=0.092,
            kelly_fraction=0.108,
            extra={},
        )
    ]


@pytest.fixture
def sample_risk_summary(sample_opportunities: list[Opportunity]) -> RiskSummary:
    position = PortfolioPosition(opportunity=sample_opportunities[0], stake=50.0)
    return RiskSummary(
        bankroll=1000.0,
        opportunity_fraction=0.05,
        portfolio_fraction=0.1,
        positions=(position,),
        exposure_by_event={("KC@BUF", "moneyline"): 50.0},
        correlation_exposure={"KC": 50.0},
        simulation=analytics_module.BankrollSimulationResult(
            [
                analytics_module.BankrollTrajectory([1000.0, 1040.0, 1080.0]),
                analytics_module.BankrollTrajectory([1000.0, 980.0, 960.0]),
            ]
        ),
    )


@pytest.fixture
def sample_simulation() -> list[SimulationResult]:
    return [
        SimulationResult(
            event_id="KC@BUF",
            home_team="Buffalo Bills",
            away_team="Kansas City Chiefs",
            iterations=1000,
            home_win_probability=0.44,
            away_win_probability=0.52,
            expected_margin=-2.1,
            expected_total=47.3,
            margin_distribution={0: 60, 3: 120},
            total_distribution={44: 220, 51: 80},
            home_score_distribution={21: 200},
            away_score_distribution={24: 240},
        )
    ]


def test_ladder_matrix(sample_quotes: list[IngestedOdds]) -> None:
    ladders = build_ladder_matrix(sample_quotes)
    assert ("KC@BUF", "spread", "Buffalo Bills") in ladders
    matrix = ladders[("KC@BUF", "spread", "Buffalo Bills")]
    assert "1sthalf" in matrix
    assert matrix["1sthalf"][-2.5] == -105


def test_dashboard_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    sample_quotes,
    sample_simulation,
    sample_opportunities,
    sample_risk_summary,
) -> None:
    class _FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return dt.datetime(2024, 1, 21, 13, 30, tzinfo=tz)

    monkeypatch.setattr(dashboard_module.dt, "datetime", _FixedDateTime)
    dashboard = Dashboard()
    dashboard.set_search("chiefs")
    snapshot = dashboard.snapshot(
        sample_quotes,
        sample_simulation,
        sample_opportunities,
        risk_summary=sample_risk_summary,
    )
    assert isinstance(snapshot, DashboardSnapshot)
    assert snapshot.header[0] == "NFL Terminal — 2024-01-21 13:30Z"
    risk_panel = next(view for view in snapshot.panels if view.state.key == "risk")
    assert any("Opportunity Kelly fraction" in line for line in risk_panel.body)
    assert any("Simulation drawdowns" in line for line in risk_panel.body)
    assert any("Mean terminal" in line for line in risk_panel.body)
    output = dashboard.render(
        sample_quotes,
        sample_simulation,
        sample_opportunities,
        risk_summary=sample_risk_summary,
    )
    assert "Search Results" in output
    assert "Kansas City Chiefs" in output
    assert "1.83" in output
    assert "5/6" in output


def test_terminal_session_commands(sample_quotes, sample_simulation, sample_opportunities) -> None:
    session = TerminalDashboardSession()
    response = session.handle("filter sportsbooks=FanDuel", sample_quotes, sample_simulation, sample_opportunities)
    assert response == "Filters updated."
    session.handle("search Chiefs", sample_quotes, sample_simulation, sample_opportunities)
    rendered = session.handle("show", sample_quotes, sample_simulation, sample_opportunities)
    assert "Kansas City Chiefs" in rendered
    cleared = session.handle("clear search", sample_quotes, sample_simulation, sample_opportunities)
    assert cleared == "Search cleared."


def test_keyboard_controller(sample_quotes, sample_simulation, sample_opportunities, sample_risk_summary) -> None:
    controller = DashboardKeyboardController()
    snapshot = controller.refresh(
        sample_quotes,
        sample_simulation,
        sample_opportunities,
        risk_summary=sample_risk_summary,
    )
    assert isinstance(snapshot, DashboardSnapshot)
    assert controller.current_panel_key == controller.panel_keys[0]
    next_key = controller.focus_next_panel()
    assert next_key == controller.panel_keys[1]
    controller.toggle_current_panel()
    assert controller.dashboard._panels[next_key].collapsed is True
    controller.toggle_current_panel()
    assert controller.dashboard._panels[next_key].collapsed is False
    controller.toggle_quarters()
    assert controller.dashboard.filters.include_quarters is False
    controller.toggle_halves()
    assert controller.dashboard.filters.include_halves is False
    controller.apply_filter_expression("sportsbooks=FanDuel")
    assert controller.dashboard.filters.sportsbooks == frozenset({"FanDuel"})
    controller.apply_filter_expression("")
    assert controller.dashboard.filters.sportsbooks is None
    controller.apply_search("Chiefs")
    assert controller.dashboard.search.query == "Chiefs"
    controller.clear_search()
    assert not controller.dashboard.search.query
