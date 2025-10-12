from __future__ import annotations

import datetime as dt

import pytest

from nflreadpy.betting.dashboard import Dashboard, TerminalDashboardSession, _build_ladder_matrix
from nflreadpy.betting import dashboard as dashboard_module
from nflreadpy.betting.analytics import Opportunity
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
    ladders = _build_ladder_matrix(sample_quotes)
    assert ("KC@BUF", "spread", "Buffalo Bills") in ladders
    matrix = ladders[("KC@BUF", "spread", "Buffalo Bills")]
    assert "1sthalf" in matrix
    assert matrix["1sthalf"][-2.5] == -105


def test_dashboard_snapshot(monkeypatch: pytest.MonkeyPatch, sample_quotes, sample_simulation, sample_opportunities) -> None:
    class _FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return dt.datetime(2024, 1, 21, 13, 30, tzinfo=tz)

    monkeypatch.setattr(dashboard_module.dt, "datetime", _FixedDateTime)
    dashboard = Dashboard()
    dashboard.set_search("chiefs")
    output = dashboard.render(sample_quotes, sample_simulation, sample_opportunities)
    assert "NFL Terminal — 2024-01-21 13:30Z" in output
    assert "Search Results" in output
    assert "Kansas City Chiefs" in output


def test_terminal_session_commands(sample_quotes, sample_simulation, sample_opportunities) -> None:
    session = TerminalDashboardSession()
    response = session.handle("filter sportsbooks=FanDuel", sample_quotes, sample_simulation, sample_opportunities)
    assert response == "Filters updated."
    session.handle("search Chiefs", sample_quotes, sample_simulation, sample_opportunities)
    rendered = session.handle("show", sample_quotes, sample_simulation, sample_opportunities)
    assert "Kansas City Chiefs" in rendered
    cleared = session.handle("clear search", sample_quotes, sample_simulation, sample_opportunities)
    assert cleared == "Search cleared."
