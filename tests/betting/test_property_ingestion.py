"""Hypothesis-based tests for ingestion validation edge cases."""

from __future__ import annotations

import datetime as dt
import tempfile
import uuid
from pathlib import Path

from hypothesis import given, strategies as st

from nflreadpy.betting.ingestion import OddsIngestionService
from nflreadpy.betting.scrapers.base import OddsQuote


def _odds_quote(observed_at: dt.datetime) -> OddsQuote:
    return OddsQuote(
        event_id="event",
        sportsbook="book",
        book_market_group="group",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="team",
        side=None,
        line=None,
        american_odds=110,
        observed_at=observed_at,
        extra={"sample": "payload"},
    )


@given(
    stale_seconds=st.integers(min_value=1, max_value=90 * 60),
    extra_seconds=st.integers(min_value=1, max_value=2 * 90 * 60),
    naive_timestamp=st.booleans(),
)
def test_validate_quote_flags_stale(
    stale_seconds: int, extra_seconds: int, naive_timestamp: bool
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / f"stale-{uuid.uuid4().hex}.sqlite3"
        service = OddsIngestionService(
            scrapers=[],
            storage_path=storage_path,
            stale_after=dt.timedelta(seconds=stale_seconds),
        )
        reference_time = dt.datetime.now(dt.timezone.utc)
        observed_at = reference_time - dt.timedelta(seconds=stale_seconds + extra_seconds)
        if naive_timestamp:
            observed_at = observed_at.replace(tzinfo=None)
        quote = _odds_quote(observed_at)
        reason, details = service._validate_quote(quote, reference_time)
        assert reason == "stale"
        assert details == []


class _StubComplianceEngine:
    def __init__(self, compliant: bool, reasons: list[str]):
        self.compliant = compliant
        self.reasons = reasons
        self.calls: list[dict[str, object]] = []

    def evaluate_metadata(
        self,
        *,
        sportsbook: str,
        market: str,
        event_id: str,
        metadata,
        log: bool,
    ) -> tuple[bool, list[str]]:
        self.calls.append(
            {
                "sportsbook": sportsbook,
                "market": market,
                "event_id": event_id,
                "metadata": metadata,
                "log": log,
            }
        )
        return self.compliant, list(self.reasons)


@given(
    compliant=st.booleans(),
    reasons=st.lists(
        st.text(min_size=1, max_size=12, alphabet=st.characters(min_codepoint=33, max_codepoint=126)),
        max_size=3,
    ),
)
def test_validate_quote_respects_compliance(
    compliant: bool, reasons: list[str]
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / f"compliance-{uuid.uuid4().hex}.sqlite3"
        engine = _StubComplianceEngine(compliant, reasons)
        service = OddsIngestionService(
            scrapers=[],
            storage_path=storage_path,
            stale_after=dt.timedelta(minutes=5),
            compliance_engine=engine,
        )
        reference_time = dt.datetime.now(dt.timezone.utc)
        quote = _odds_quote(reference_time)
        reason, details = service._validate_quote(quote, reference_time)
        if compliant:
            assert reason is None
            assert details == []
        else:
            assert reason is not None
            assert reason.startswith("non_compliant")
            assert details == reasons
        assert engine.calls, "Compliance engine should be consulted"
