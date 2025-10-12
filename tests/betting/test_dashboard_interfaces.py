from __future__ import annotations

import datetime as dt
import sys
from types import ModuleType

import pytest

try:  # pragma: no cover - optional dependency for dashboard tables
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - fallback when polars absent
    pl = None  # type: ignore[assignment]

from nflreadpy.betting.analytics import Opportunity
from nflreadpy.betting.dashboard import Dashboard
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.models import SimulationResult
from nflreadpy.betting.web import (
    CalibrationPoint,
    LineMovementPoint,
    PortfolioPosition,
    run_dashboard,
)
from nflreadpy.betting.web.app import (
    _calibration_frame,
    _ladder_frame,
    _line_history_frame,
    _live_market_table,
    _opportunities_table,
    _portfolio_table,
)


@pytest.fixture
def sample_quotes() -> list[IngestedOdds]:
    base_time = dt.datetime(2024, 1, 7, 12, 0, tzinfo=dt.timezone.utc)
    return [
        IngestedOdds(
            event_id="BUF@KC",
            sportsbook="BookA",
            book_market_group="moneyline",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="BUF",
            side=None,
            line=None,
            american_odds=-120,
            observed_at=base_time,
            extra={},
        ),
        IngestedOdds(
            event_id="BUF@KC",
            sportsbook="BookA",
            book_market_group="spread",
            market="spread",
            scope="1Q",
            entity_type="team",
            team_or_player="BUF",
            side="home",
            line=-0.5,
            american_odds=-105,
            observed_at=base_time + dt.timedelta(minutes=5),
            extra={},
        ),
        IngestedOdds(
            event_id="BUF@KC",
            sportsbook="BookB",
            book_market_group="spread",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="KC",
            side="away",
            line=2.5,
            american_odds=110,
            observed_at=base_time + dt.timedelta(minutes=10),
            extra={},
        ),
    ]


@pytest.fixture
def sample_simulations() -> list[SimulationResult]:
    return [
        SimulationResult(
            event_id="BUF@KC",
            home_team="KC",
            away_team="BUF",
            iterations=1000,
            home_win_probability=0.48,
            away_win_probability=0.5,
            expected_margin=-1.5,
            expected_total=48.1,
            margin_distribution={-3: 300, -1: 200, 0: 100, 3: 400},
            total_distribution={42: 200, 45: 300, 48: 200, 51: 300},
            home_score_distribution={21: 250, 24: 250, 27: 200, 30: 300},
            away_score_distribution={21: 200, 24: 300, 27: 250, 30: 250},
        )
    ]


@pytest.fixture
def sample_opportunities() -> list[Opportunity]:
    return [
        Opportunity(
            event_id="BUF@KC",
            sportsbook="BookB",
            book_market_group="spread",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="KC",
            side="away",
            line=2.5,
            american_odds=110,
            model_probability=0.55,
            push_probability=0.02,
            implied_probability=0.476,
            expected_value=0.07,
            kelly_fraction=0.05,
            extra={},
        )
    ]


def test_dashboard_renders_filters_and_ladders(sample_quotes, sample_simulations, sample_opportunities):
    dashboard = Dashboard()
    output = dashboard.render(sample_quotes, sample_simulations, sample_opportunities)
    assert "Active Filters:" in output
    assert "Line Ladders" in output
    assert "BUF@KC" in output


def test_dashboard_toggle_quarters_filters_out_scope(sample_quotes, sample_simulations, sample_opportunities):
    dashboard = Dashboard()
    dashboard.toggle_quarters()
    result = dashboard.render(sample_quotes, sample_simulations, sample_opportunities)
    assert "1Q" not in result
    assert "spread" in result


def test_dashboard_panel_collapse(sample_quotes, sample_simulations, sample_opportunities):
    dashboard = Dashboard()
    dashboard.toggle_panel("ladders")
    result = dashboard.render(sample_quotes, sample_simulations, sample_opportunities)
    assert "Line Ladders (collapsed)" in result


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_live_market_table_structure(sample_quotes):
    table = _live_market_table(sample_quotes, stale_after=dt.timedelta(minutes=5))
    assert table.columns == [
        "event_id",
        "sportsbook",
        "market",
        "scope",
        "selection",
        "side",
        "line",
        "american_odds",
        "age",
        "fresh",
        "observed_at",
    ]
    assert table.height == 3


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_line_history_frame_orders_points():
    base_time = dt.datetime(2024, 1, 1, 15, 0, tzinfo=dt.timezone.utc)
    frame = _line_history_frame(
        [
            LineMovementPoint(
                event_id="BUF@KC",
                sportsbook="BookA",
                market="spread",
                scope="game",
                selection="BUF",
                american_odds=-110,
                line=-2.5,
                observed_at=base_time + dt.timedelta(minutes=10),
            ),
            LineMovementPoint(
                event_id="BUF@KC",
                sportsbook="BookA",
                market="spread",
                scope="game",
                selection="BUF",
                american_odds=-115,
                line=-3.0,
                observed_at=base_time,
            ),
        ]
    )
    assert frame["observed_at"].to_list() == [
        (base_time).isoformat(),
        (base_time + dt.timedelta(minutes=10)).isoformat(),
    ]


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_portfolio_table_totals(sample_quotes):
    positions = [
        PortfolioPosition(
            event_id="BUF@KC",
            sportsbook="BookA",
            market="moneyline",
            scope="game",
            selection="BUF",
            stake=100.0,
            price=-120,
            status="open",
            expected_value=0.05,
            kelly_fraction=0.02,
            pnl=0.0,
        ),
        PortfolioPosition(
            event_id="BUF@KC",
            sportsbook="BookB",
            market="spread",
            scope="game",
            selection="KC +2.5",
            stake=75.0,
            price=110,
            status="settled",
            expected_value=0.08,
            kelly_fraction=0.03,
            pnl=82.5,
        ),
    ]
    table = _portfolio_table(positions)
    totals = table.select(
        [
            pl.col("stake").sum().alias("total_stake"),
            pl.col("pnl").sum().alias("realized_pnl"),
        ]
    )
    assert totals.row(0) == (175.0, 82.5)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_ladder_frame_marks_best_scope():
    ladder = {
        "1sthalf": {-2.5: -105},
        "game": {-2.5: -102},
    }
    frame = _ladder_frame(ladder)
    assert frame.columns == ["line", "1sthalf", "game", "best_scope"]
    assert frame.height == 1
    assert frame["best_scope"].to_list() == ["game"]


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_opportunities_table_columns(sample_opportunities):
    table = _opportunities_table(sample_opportunities)
    assert table.columns == [
        "event_id",
        "sportsbook",
        "market",
        "scope",
        "selection",
        "american_odds",
        "model_probability",
        "expected_value",
        "kelly_fraction",
    ]
    assert table.height == 1


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_calibration_frame_structure():
    frame = _calibration_frame(
        [
            CalibrationPoint(
                market="moneyline",
                bucket="0.5-0.6",
                expected=0.55,
                observed=0.5,
                sample_size=100,
            ),
            CalibrationPoint(
                market="spread",
                bucket="0.6-0.7",
                expected=0.65,
                observed=0.66,
                sample_size=40,
            ),
        ]
    )
    assert frame.height == 2
    assert frame["market"].to_list() == ["moneyline", "spread"]


class _FakeSidebar:
    def __init__(self) -> None:
        self.headers: list[str] = []
        self.multiselect_calls: list[tuple[str, list[str], list[str]]] = []
        self.checkbox_calls: list[tuple[str, bool]] = []
        self.slider_calls: list[tuple[str, object]] = []

    def header(self, text: str) -> None:
        self.headers.append(text)

    def multiselect(self, label: str, options: list[str], default: list[str]):
        self.multiselect_calls.append((label, options, default))
        return default

    def checkbox(self, label: str, value: bool = True) -> bool:
        self.checkbox_calls.append((label, value))
        return value

    def slider(
        self,
        label: str,
        min_value=None,
        max_value=None,
        value=None,
        step: int | None = None,
    ):
        self.slider_calls.append((label, value))
        return value


class _FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.sidebar = _FakeSidebar()
        self.headers: list[str] = []
        self.captions: list[str] = []
        self.dataframes: list[tuple[object, bool]] = []
        self.info_messages: list[str] = []
        self.vega_specs: list[dict] = []
        self.writes: list[str] = []

    def set_page_config(self, **kwargs) -> None:  # pragma: no cover - trivial
        self.page_config = kwargs

    def title(self, text: str) -> None:
        self.headers.append(text)

    def header(self, text: str) -> None:
        self.headers.append(text)

    def subheader(self, text: str) -> None:
        self.headers.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def info(self, text: str) -> None:
        self.info_messages.append(text)

    def dataframe(self, data, use_container_width: bool = False) -> None:
        self.dataframes.append((data, use_container_width))

    def vega_lite_chart(self, spec: dict, use_container_width: bool = False) -> None:
        self.vega_specs.append(spec)

    def write(self, text: str) -> None:
        self.writes.append(text)


class _StubProvider:
    def __init__(self, quotes, opportunities):
        self._quotes = quotes
        self._opps = opportunities

    def live_markets(self):
        return self._quotes

    def opportunities(self):
        return self._opps

    def line_history(self):
        base = dt.datetime(2024, 1, 7, 11, 0, tzinfo=dt.timezone.utc)
        return [
            LineMovementPoint(
                event_id="BUF@KC",
                sportsbook="BookA",
                market="spread",
                scope="game",
                selection="BUF",
                american_odds=-110,
                line=-2.5,
                observed_at=base,
            ),
            LineMovementPoint(
                event_id="BUF@KC",
                sportsbook="BookB",
                market="spread",
                scope="game",
                selection="KC",
                american_odds=105,
                line=2.5,
                observed_at=base + dt.timedelta(minutes=30),
            ),
        ]

    def calibration(self):
        return [
            CalibrationPoint(
                market="moneyline",
                bucket="0.5-0.6",
                expected=0.55,
                observed=0.53,
                sample_size=120,
            )
        ]

    def portfolio(self):
        return [
            PortfolioPosition(
                event_id="BUF@KC",
                sportsbook="BookA",
                market="moneyline",
                scope="game",
                selection="BUF",
                stake=100.0,
                price=-120,
                status="open",
                expected_value=0.05,
                kelly_fraction=0.02,
                pnl=0.0,
            )
        ]


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_run_dashboard_uses_streamlit(monkeypatch, sample_quotes, sample_opportunities):
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    provider = _StubProvider(sample_quotes, sample_opportunities)
    run_dashboard(provider)

    assert fake_streamlit.headers[0] == "NFL Betting Dashboard"
    assert any("Live Markets" in header for header in fake_streamlit.headers)
    assert fake_streamlit.dataframes  # tables rendered
    assert fake_streamlit.vega_specs  # charts rendered
    assert fake_streamlit.writes  # portfolio summary output
