from __future__ import annotations

import datetime as dt

import pytest

from nflreadpy.betting.dashboard import DashboardHotkey, TerminalDashboardSession
from nflreadpy.betting.ingestion import IngestedOdds


@pytest.fixture
def odds_sample() -> list[IngestedOdds]:
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
        )
    ]


def test_hotkey_bindings_are_exposed() -> None:
    session = TerminalDashboardSession()
    bindings = session.dashboard.hotkey_bindings()
    assert any(binding.key == "q" for binding in bindings)
    assert isinstance(bindings[0], DashboardHotkey)


def test_hotkey_dispatch_aliases_commands(
    odds_sample: list[IngestedOdds],
) -> None:
    session = TerminalDashboardSession()
    response = session.handle("hotkey q", odds_sample, [], [])
    assert "Quarter markets" in response
    response = session.handle("hotkey / Chiefs", odds_sample, [], [])
    assert session.dashboard.search.query == "Chiefs"
    assert "Search set" in response
    response = session.handle("hotkey l", odds_sample, [], [])
    assert "Panel 'ladders'" in response


def test_hotkeys_command_lists_bindings(odds_sample: list[IngestedOdds]) -> None:
    session = TerminalDashboardSession()
    response = session.handle("hotkeys", odds_sample, [], [])
    assert "Hotkey bindings" in response
    assert "q -> toggle quarters" in response


def test_unknown_hotkey_is_reported(odds_sample: list[IngestedOdds]) -> None:
    session = TerminalDashboardSession()
    response = session.handle("hotkey z", odds_sample, [], [])
    assert "Unknown hotkey 'z'" in response


def test_hotkey_delegates_arguments(
    odds_sample: list[IngestedOdds],
) -> None:
    session = TerminalDashboardSession()
    response = session.handle("hotkey f sportsbooks=FanDuel", odds_sample, [], [])
    assert response == "Filters updated."
    assert session.dashboard.filters.sportsbooks == frozenset({"FanDuel"})
