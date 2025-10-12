from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from nflreadpy.betting.dashboard_core import (
    DashboardFilters,
    DEFAULT_SEARCH_TARGETS,
    build_ladder_matrix,
    is_half_scope,
    is_quarter_scope,
    normalize_scope,
    parse_filter_tokens,
)
from nflreadpy.betting.ingestion import IngestedOdds


def test_parse_filter_tokens_handles_multiple_values() -> None:
    tokens = parse_filter_tokens(["sportsbooks=FanDuel,DraftKings", "markets=moneyline"])
    assert tokens["sportsbooks"] == ["FanDuel", "DraftKings"]
    assert tokens["markets"] == ["moneyline"]


def test_dashboard_filters_description_and_match(sample_quote: IngestedOdds) -> None:
    filters = DashboardFilters().update(sportsbooks=[sample_quote.sportsbook], include_quarters=False)
    description = filters.description()
    assert any(sample_quote.sportsbook in piece for piece in description)
    assert any("Quarters=off" in piece for piece in description)
    assert filters.match_odds(sample_quote)


def test_scope_normalisation_helpers() -> None:
    assert normalize_scope("First Quarter") == "firstquarter"
    assert is_quarter_scope("1q")
    assert is_quarter_scope(normalize_scope("Q3"))
    assert is_half_scope(normalize_scope("1st Half"))
    assert not is_half_scope("game")


def test_build_ladder_matrix_orders_lines(sample_quote: IngestedOdds) -> None:
    quote_with_line = dataclasses.replace(
        sample_quote, line=-2.5, american_odds=-110, scope="1st Half"
    )
    ladder = build_ladder_matrix([quote_with_line])
    key = (quote_with_line.event_id, quote_with_line.market, quote_with_line.team_or_player)
    assert key in ladder
    scope_key = normalize_scope("1st Half")
    assert ladder[key][scope_key][-2.5] == -110


@pytest.fixture
def sample_quote() -> IngestedOdds:
    return IngestedOdds(
        event_id="KC@BUF",
        sportsbook="FanDuel",
        book_market_group="moneyline",
        market="moneyline",
        scope="Game",
        entity_type="team",
        team_or_player="Kansas City Chiefs",
        side=None,
        line=None,
        american_odds=-120,
        observed_at=dt.datetime(2024, 1, 21, 12, 0, tzinfo=dt.timezone.utc),
        extra={},
    )


def test_default_search_targets_is_frozen() -> None:
    assert isinstance(DEFAULT_SEARCH_TARGETS, frozenset)
    with pytest.raises(AttributeError):
        DEFAULT_SEARCH_TARGETS.add("new")  # type: ignore[attr-defined]
