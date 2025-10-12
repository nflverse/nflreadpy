from __future__ import annotations

import datetime as dt

import pytest

from nflreadpy.betting.analytics import Opportunity
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.web import (
    CalibrationPoint,
    LineMovementPoint,
    PortfolioPosition,
    create_api_app,
)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402  # isort:skip


class DummyProvider:
    def live_markets(self):
        observed = dt.datetime(2024, 1, 21, 15, 0, tzinfo=dt.timezone.utc)
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

    def opportunities(self):
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

    def line_history(self):
        observed = dt.datetime(2024, 1, 21, 14, 15, tzinfo=dt.timezone.utc)
        return [
            LineMovementPoint(
                event_id="KC@BUF",
                sportsbook="FanDuel",
                market="moneyline",
                scope="game",
                selection="Kansas City Chiefs",
                american_odds=-125,
                line=None,
                observed_at=observed,
            )
        ]

    def calibration(self):
        return [
            CalibrationPoint(
                market="moneyline",
                bucket="0.55-0.60",
                expected=0.58,
                observed=0.56,
                sample_size=110,
            )
        ]

    def portfolio(self):
        return [
            PortfolioPosition(
                event_id="KC@BUF",
                sportsbook="FanDuel",
                market="moneyline",
                scope="game",
                selection="Kansas City Chiefs",
                stake=100.0,
                price=-120,
                status="open",
                expected_value=0.09,
                kelly_fraction=0.11,
                pnl=None,
            )
        ]


def test_api_endpoints() -> None:
    provider = DummyProvider()
    app = create_api_app(provider)
    client = TestClient(app)

    markets = client.get("/markets").json()
    assert markets[0]["event_id"] == "KC@BUF"
    filters = client.get("/filters").json()
    assert "FanDuel" in filters["sportsbooks"]
    opportunities = client.get("/opportunities").json()
    assert opportunities[0]["expected_value"] == pytest.approx(0.092)
