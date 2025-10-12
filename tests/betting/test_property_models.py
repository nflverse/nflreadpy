"""Property-based regression tests for betting analytics primitives."""

from __future__ import annotations

import datetime as dt
import math
from typing import List

from hypothesis import given, strategies as st

from nflreadpy.betting.analytics import EdgeDetector, KellyCriterion
from nflreadpy.betting.dashboard_core import build_ladder_matrix, normalize_scope
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.models import ProbabilityTriple
from nflreadpy.betting.scrapers.base import OddsQuote
from nflreadpy.betting.utils import decimal_to_american


def _valid_american_odds() -> st.SearchStrategy[int]:
    positive = st.integers(min_value=100, max_value=1000)
    negative = st.integers(min_value=-1000, max_value=-100)
    return st.one_of(positive, negative)


@st.composite
def _probability_triples(draw: st.DrawFn) -> ProbabilityTriple:
    win = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    remaining = max(0.0, 1.0 - win)
    push = draw(
        st.floats(min_value=0.0, max_value=remaining, allow_nan=False, allow_infinity=False)
    )
    return ProbabilityTriple(win=win, push=push)


@st.composite
def _quote_pairs(draw: st.DrawFn) -> List[OddsQuote]:
    base_decimal = draw(
        st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    delta = draw(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    richer_decimal = min(15.0, base_decimal + delta)
    american_low = decimal_to_american(base_decimal)
    american_high = decimal_to_american(richer_decimal)
    quote_low = OddsQuote(
        event_id="event",
        sportsbook="book",
        book_market_group="market_group",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="team",
        side=None,
        line=None,
        american_odds=american_low,
        observed_at=dt.datetime.now(dt.timezone.utc),
    )
    quote_high = OddsQuote(
        event_id="event",
        sportsbook="book",
        book_market_group="market_group",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="team",
        side=None,
        line=None,
        american_odds=american_high,
        observed_at=quote_low.observed_at,
    )
    if quote_high.decimal_multiplier() < quote_low.decimal_multiplier():
        quote_low, quote_high = quote_high, quote_low
    return [quote_low, quote_high]


@given(probability=_probability_triples(), quotes=_quote_pairs())
def test_expected_value_monotonic_with_price(probability: ProbabilityTriple, quotes: List[OddsQuote]) -> None:
    detector = EdgeDetector(value_threshold=0.0)
    evaluations = [(quote, probability, None) for quote in quotes]
    _, expected_values = detector._evaluate_probabilities_python(evaluations)
    assert expected_values[0] <= expected_values[1] + 1e-9


@given(
    win=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    push=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    fractional_kelly=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    cap=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    price=_valid_american_odds(),
)
def test_kelly_fraction_within_unit_interval(
    win: float, push: float, fractional_kelly: float, cap: float | None, price: int
) -> None:
    push = min(push, max(0.0, 1.0 - win))
    loss = max(0.0, 1.0 - win - push)
    value = KellyCriterion.fraction(
        win,
        loss,
        price,
        fractional_kelly=fractional_kelly,
        cap=cap,
    )
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0 + 1e-9
    if cap is not None:
        assert value <= cap + 1e-9


@st.composite
def _ingested_odds(draw: st.DrawFn) -> IngestedOdds:
    scope = draw(
        st.sampled_from(["game", "1h", "2h", "First Half", "Second Half", "4q"])
    )
    line = draw(
        st.one_of(
            st.none(),
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
    )
    american = draw(_valid_american_odds())
    return IngestedOdds(
        event_id=f"event-{draw(st.integers(min_value=0, max_value=5))}",
        sportsbook="book",
        book_market_group="group",
        market=draw(st.sampled_from(["spread", "total", "moneyline"])),
        scope=scope,
        entity_type="team",
        team_or_player=f"team-{draw(st.integers(min_value=0, max_value=3))}",
        side=draw(st.one_of(st.none(), st.sampled_from(["over", "under", "fav", "dog"]))),
        line=None if line is None else float(line),
        american_odds=american,
        observed_at=dt.datetime.now(dt.timezone.utc),
        extra={},
    )


@given(quotes=st.lists(_ingested_odds(), min_size=1, max_size=12))
def test_ladder_matrix_monotonic_lines(quotes: List[IngestedOdds]) -> None:
    ladder = build_ladder_matrix(quotes)
    for (event_id, market, selection), scopes in ladder.items():
        for scope, entries in scopes.items():
            rounded_lines = sorted(
                {
                    round(float(quote.line), 1)
                    for quote in quotes
                    if quote.line is not None
                    and quote.event_id == event_id
                    and quote.market == market
                    and quote.team_or_player == selection
                    and normalize_scope(quote.scope) == scope
                }
            )
            assert list(sorted(entries)) == rounded_lines
