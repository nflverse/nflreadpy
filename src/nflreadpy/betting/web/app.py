"""Streamlit dashboard for nflreadpy betting data."""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import polars as pl
    from ..analytics import LineMovement

from ..analytics import Opportunity, line_movement_summary
from ..dashboard import Dashboard
from ..dashboard_core import (
    DashboardFilters,
    build_movement_series,
    format_age,
    is_half_scope,
    is_quarter_scope,
    normalize_scope,
    build_ladder_matrix,
    render_sparkline,
)
from ..ingestion import IngestedOdds
from ..utils import american_to_decimal


@dataclasses.dataclass(slots=True)
class LineMovementPoint:
    """Historical price for a single selection at a specific timestamp."""

    event_id: str
    sportsbook: str
    market: str
    scope: str
    selection: str
    american_odds: int
    line: float | None
    observed_at: dt.datetime


@dataclasses.dataclass(slots=True)
class CalibrationPoint:
    """Calibration summary for a probability bucket."""

    market: str
    bucket: str
    expected: float
    observed: float
    sample_size: int


@dataclasses.dataclass(slots=True)
class PortfolioPosition:
    """Represents an entry in the bettor's portfolio view."""

    event_id: str
    sportsbook: str
    market: str
    scope: str
    selection: str
    stake: float
    price: int
    status: str
    expected_value: float | None = None
    kelly_fraction: float | None = None
    pnl: float | None = None


class DashboardDataProvider(Protocol):
    """Provider interface consumed by :func:`run_dashboard`."""

    def live_markets(self) -> Sequence[IngestedOdds]:
        ...

    def opportunities(self) -> Sequence[Opportunity]:  # pragma: no cover - optional
        return []

    def line_history(self) -> Sequence[LineMovementPoint]:
        ...

    def calibration(self) -> Sequence[CalibrationPoint]:
        ...

    def portfolio(self) -> Sequence[PortfolioPosition]:
        ...


def run_dashboard(provider: DashboardDataProvider, *, title: str = "NFL Betting Dashboard") -> None:
    """Render the Streamlit dashboard using a :class:`DashboardDataProvider`."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised indirectly
        raise RuntimeError(
            "Streamlit is required to launch the web dashboard. Install with 'uv add streamlit'."
        ) from exc

    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    sidebar = st.sidebar
    sidebar.header("Session")

    refresh_seconds = 15
    slider = getattr(sidebar, "slider", None)
    if callable(slider):
        refresh_seconds = slider(
            "Auto-refresh seconds", min_value=5, max_value=60, value=refresh_seconds, step=5
        )

    refresher = getattr(st, "autorefresh", None)
    sidebar_caption = getattr(sidebar, "caption", getattr(st, "caption", lambda *_args, **_kwargs: None))
    if callable(refresher):  # pragma: no branch - depends on streamlit version
        refresher(interval=int(refresh_seconds * 1000), key="nfl-dashboard-refresh")
    else:  # pragma: no cover - informative message only when feature absent
        sidebar_caption("Install Streamlit 1.27+ for automatic refresh support.")

    button = getattr(sidebar, "button", None)
    if callable(button) and button("Refresh now"):
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):  # pragma: no branch - streamlit shim
            rerun()

    dashboard = Dashboard()
    live_quotes = list(provider.live_markets())
    opportunities = list(getattr(provider, "opportunities", lambda: [])())
    filters = DashboardFilters()
    available = dashboard.available_options(live_quotes, opportunities)

    sidebar.header("Filters")
    selected_books = sidebar.multiselect(
        "Sportsbooks", available["sportsbooks"], default=available["sportsbooks"]
    )
    selected_groups = sidebar.multiselect(
        "Market groups", available["market_groups"], default=available["market_groups"]
    )
    selected_markets = sidebar.multiselect(
        "Markets", available["markets"], default=available["markets"]
    )
    selected_scopes = sidebar.multiselect(
        "Scopes", available["scopes"], default=available["scopes"]
    )
    selected_events = sidebar.multiselect(
        "Events", available["events"], default=available["events"]
    )
    include_quarters = sidebar.checkbox("Include quarter markets", value=True)
    include_halves = sidebar.checkbox("Include half markets", value=True)
    default_depth = dashboard.movement_depth or 6
    movement_depth = sidebar.slider(
        "Movement depth", min_value=1, max_value=20, value=default_depth
    )
    default_stale_minutes = int(dashboard.stale_after.total_seconds() // 60)
    stale_after_minutes = sidebar.slider(
        "Stale after (minutes)", min_value=1, max_value=60, value=default_stale_minutes
    )
    stale_after = dt.timedelta(minutes=stale_after_minutes)

    filters = filters.update(
        sportsbooks=_fallback_to_all(selected_books, available["sportsbooks"]),
        market_groups=_fallback_to_all(selected_groups, available["market_groups"]),
        markets=_fallback_to_all(selected_markets, available["markets"]),
        scopes=_fallback_to_all(selected_scopes, available["scopes"]),
        events=_fallback_to_all(selected_events, available["events"]),
        include_quarters=include_quarters,
        include_halves=include_halves,
    )

    filtered_quotes = [quote for quote in live_quotes if filters.match_odds(quote)]
    filtered_opportunities = [opp for opp in opportunities if filters.match_opportunity(opp)]

    st.caption("Filters applied: " + " | ".join(filters.description()))

    st.caption(f"Last updated: {dt.datetime.now(dt.timezone.utc).isoformat()}")

    st.header("Live Markets")
    live_table = _live_market_table(filtered_quotes, stale_after=stale_after)
    if live_table.height == 0:
        st.info("No live markets available for the selected filters.")
    else:
        st.dataframe(_as_streamlit_data(live_table), use_container_width=True)

    if filtered_opportunities:
        st.subheader("Model Opportunities")
        opp_table = _opportunities_table(filtered_opportunities)
        st.dataframe(_as_streamlit_data(opp_table), use_container_width=True)

    st.header("Line Movement")
    raw_history = [
        point for point in provider.line_history() if _line_point_matches(point, filters)
    ]
    history_quotes = _line_history_quotes(raw_history)
    movements = line_movement_summary(history_quotes, depth=movement_depth)
    movement_series = build_movement_series(
        history_quotes,
        max_points=max(movement_depth * 4, 12),
    )
    top_keys = {movement.key for movement in movements}
    history_points = (
        [point for point in raw_history if _line_point_key(point) in top_keys]
        if top_keys
        else raw_history
    )
    history_frame = _line_history_frame(history_points)
    if history_frame.height == 0:
        st.info("No line movement history for the selected filters.")
    else:
        st.vega_lite_chart(
            {
                "data": {"values": history_frame.to_dicts()},
                "mark": {"type": "line"},
                "encoding": {
                    "x": {"field": "observed_at", "type": "temporal", "title": "Timestamp"},
                    "y": {
                        "field": "american_odds",
                        "type": "quantitative",
                        "title": "American odds",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "title": "Selection",
                    },
                    "tooltip": [
                        {"field": "series", "type": "nominal"},
                        {"field": "observed_at", "type": "temporal"},
                        {"field": "american_odds", "type": "quantitative"},
                        {"field": "line", "type": "quantitative"},
                    ],
                },
            },
            use_container_width=True,
        )
    summary_frame = _movement_summary_frame(movements, movement_series)
    if summary_frame.height == 0:
        st.info("No significant movement detected within the selected depth.")
    else:
        st.dataframe(_as_streamlit_data(summary_frame), use_container_width=True)

    st.header("Line Ladders")
    ladder_matrix = build_ladder_matrix(filtered_quotes)
    if not ladder_matrix:
        st.info("No ladder matrices available for the selected filters.")
    else:
        for key, ladder in sorted(ladder_matrix.items()):
            event_id, market, selection = key
            st.subheader(f"{event_id} — {market} — {selection}")
            ladder_frame = _ladder_frame(ladder)
            st.dataframe(_as_streamlit_data(ladder_frame), use_container_width=True)

    st.header("Calibration")
    calibration_points = [
        point for point in provider.calibration() if _calibration_matches(point, filters)
    ]
    calibration_frame = _calibration_frame(calibration_points)
    if calibration_frame.height == 0:
        st.info("No calibration diagnostics available for the selected filters.")
    else:
        st.vega_lite_chart(
            {
                "data": {"values": calibration_frame.to_dicts()},
                "mark": {"type": "point", "filled": True, "size": 80},
                "encoding": {
                    "x": {
                        "field": "expected",
                        "type": "quantitative",
                        "title": "Model probability",
                    },
                    "y": {
                        "field": "observed",
                        "type": "quantitative",
                        "title": "Observed frequency",
                    },
                    "color": {"field": "market", "type": "nominal", "title": "Market"},
                    "size": {"field": "sample_size", "type": "quantitative", "title": "Samples"},
                    "tooltip": [
                        {"field": "market", "type": "nominal"},
                        {"field": "bucket", "type": "nominal"},
                        {"field": "expected", "type": "quantitative"},
                        {"field": "observed", "type": "quantitative"},
                        {"field": "sample_size", "type": "quantitative"},
                    ],
                },
            },
            use_container_width=True,
        )

    st.header("Portfolio")
    positions = [position for position in provider.portfolio() if _position_matches(position, filters)]
    portfolio_table = _portfolio_table(positions)
    if portfolio_table.height == 0:
        st.info("No portfolio positions for the selected filters.")
    else:
        total_stake = float(portfolio_table["stake"].fill_null(0.0).sum())
        realized_pnl = float(portfolio_table["pnl"].fill_null(0.0).sum())
        st.write(
            f"Total stake: ${total_stake:,.2f} — Realized PnL: ${realized_pnl:,.2f}"
        )
        st.dataframe(_as_streamlit_data(portfolio_table), use_container_width=True)
def _fallback_to_all(selected: Sequence[str], available: Sequence[str]) -> Iterable[str] | None:
    if not selected or len(selected) == len(available):
        return None
    return selected


def _live_market_table(
    odds: Sequence[IngestedOdds], *, stale_after: dt.timedelta
) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not odds:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "sportsbook": pl.String,
                "market": pl.String,
                "scope": pl.String,
                "selection": pl.String,
                "side": pl.String,
                "line": pl.Float64,
                "american_odds": pl.Int64,
                "age": pl.String,
                "fresh": pl.String,
                "observed_at": pl.String,
            }
        )
    now = dt.datetime.now(dt.timezone.utc)
    return pl.DataFrame(
        [
            {
                "event_id": quote.event_id,
                "sportsbook": quote.sportsbook,
                "market": quote.market,
                "scope": quote.scope,
                "selection": quote.team_or_player,
                "side": quote.side or "-",
                "line": quote.line,
                "american_odds": quote.american_odds,
                "age": format_age(now - quote.observed_at),
                "fresh": "⚠️ Stale"
                if (now - quote.observed_at) > stale_after
                else "Fresh",
                "observed_at": quote.observed_at.isoformat(),
            }
            for quote in odds
        ]
    ).sort(["event_id", "market", "selection", "sportsbook"], descending=False)

def _opportunities_table(opportunities: Sequence[Opportunity]) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not opportunities:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "sportsbook": pl.String,
                "market": pl.String,
                "scope": pl.String,
                "selection": pl.String,
                "american_odds": pl.Int64,
                "model_probability": pl.Float64,
                "expected_value": pl.Float64,
                "kelly_fraction": pl.Float64,
            }
        )
    return pl.DataFrame(
        [
            {
                "event_id": opp.event_id,
                "sportsbook": opp.sportsbook,
                "market": opp.market,
                "scope": opp.scope,
                "selection": opp.team_or_player,
                "american_odds": opp.american_odds,
                "model_probability": opp.model_probability,
                "expected_value": opp.expected_value,
                "kelly_fraction": opp.kelly_fraction,
            }
            for opp in opportunities
        ]
    )


def _line_history_frame(points: Sequence[LineMovementPoint]) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not points:
        return pl.DataFrame(
            schema={
                "observed_at": pl.String,
                "american_odds": pl.Int64,
                "line": pl.Float64,
                "series": pl.String,
            }
        )
    return pl.DataFrame(
        [
            {
                "observed_at": point.observed_at.isoformat(),
                "american_odds": point.american_odds,
                "line": point.line,
                "series": f"{point.selection} ({point.sportsbook})",
            }
            for point in points
        ]
    ).sort("observed_at")


def _movement_summary_frame(
    movements: Sequence[LineMovement],
    series: Mapping[
        tuple[str, str, str, str, str | None, float | None],
        tuple[tuple[dt.datetime, int], ...],
    ],
) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not movements:
        return pl.DataFrame(
            schema={
                "event": pl.String,
                "market": pl.String,
                "scope": pl.String,
                "selection": pl.String,
                "delta": pl.Int64,
                "opening_price": pl.Int64,
                "latest_price": pl.Int64,
                "sparkline": pl.String,
            }
        )
    records: list[dict[str, object]] = []
    for movement in movements:
        history = series.get(movement.key, ())
        spark = render_sparkline([price for _, price in history])
        event_id, market, scope, selection, side, line_value = movement.key
        label = selection if side is None else f"{selection} {side}"
        if line_value is not None:
            label = f"{label} {line_value:+.1f}"
        records.append(
            {
                "event": event_id,
                "market": market,
                "scope": scope,
                "selection": label,
                "delta": movement.delta,
                "opening_price": movement.opening_price,
                "latest_price": movement.latest_price,
                "sparkline": spark,
            }
        )
    return pl.DataFrame(records)


def _calibration_frame(points: Sequence[CalibrationPoint]) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not points:
        return pl.DataFrame(
            schema={
                "market": pl.String,
                "bucket": pl.String,
                "expected": pl.Float64,
                "observed": pl.Float64,
                "sample_size": pl.Int64,
            }
        )
    return pl.DataFrame(
        [
            {
                "market": point.market,
                "bucket": point.bucket,
                "expected": point.expected,
                "observed": point.observed,
                "sample_size": point.sample_size,
            }
            for point in points
        ]
    )


def _portfolio_table(positions: Sequence[PortfolioPosition]) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not positions:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "sportsbook": pl.String,
                "market": pl.String,
                "scope": pl.String,
                "selection": pl.String,
                "stake": pl.Float64,
                "price": pl.Int64,
                "status": pl.String,
                "expected_value": pl.Float64,
                "kelly_fraction": pl.Float64,
                "pnl": pl.Float64,
            }
        )
    return pl.DataFrame(
        [
            {
                "event_id": position.event_id,
                "sportsbook": position.sportsbook,
                "market": position.market,
                "scope": position.scope,
                "selection": position.selection,
                "stake": position.stake,
                "price": position.price,
                "status": position.status,
                "expected_value": position.expected_value,
                "kelly_fraction": position.kelly_fraction,
                "pnl": position.pnl,
            }
            for position in positions
        ]
    )


def _as_streamlit_data(table):  # type: ignore[no-untyped-def]
    try:  # pragma: no cover - optional dependency
        import pandas as pd

        return table.to_pandas()
    except ModuleNotFoundError:
        return table.to_dicts()


def _lazy_polars():  # type: ignore[no-untyped-def]
    try:
        import polars as pl
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "polars is required for the betting dashboards. Install with 'uv add polars'."
        ) from exc
    return pl


def _ladder_frame(ladder: dict[str, dict[float, int]]) -> "pl.DataFrame":
    pl = _lazy_polars()
    if not ladder:
        return pl.DataFrame(schema={"line": pl.Float64})
    scopes = sorted(ladder)
    lines = sorted({line for entries in ladder.values() for line in entries})
    records: list[dict[str, object]] = []
    for line in lines:
        row: dict[str, object] = {"line": line}
        best_scope = _best_scope_for_line(ladder, line)
        for scope in scopes:
            row[scope] = ladder[scope].get(line)
        row["best_scope"] = best_scope
        records.append(row)
    return pl.DataFrame(records)


def _best_scope_for_line(ladder: dict[str, dict[float, int]], line: float) -> str | None:
    best_scope: str | None = None
    best_price: float | None = None
    for scope, entries in ladder.items():
        odds_value = entries.get(line)
        if odds_value is None:
            continue
        decimal = american_to_decimal(odds_value)
        if best_price is None or decimal > best_price:
            best_price = decimal
            best_scope = scope
    return best_scope


def _line_history_quotes(points: Sequence[LineMovementPoint]) -> list[IngestedOdds]:
    return [
        IngestedOdds(
            event_id=point.event_id,
            sportsbook=point.sportsbook,
            book_market_group=point.market,
            market=point.market,
            scope=point.scope,
            entity_type="team",
            team_or_player=point.selection,
            side=None,
            line=point.line,
            american_odds=point.american_odds,
            observed_at=point.observed_at,
            extra={},
        )
        for point in points
    ]


def _line_point_key(point: LineMovementPoint) -> tuple[str, str, str, str, str | None, float | None]:
    return (
        point.event_id,
        point.market,
        point.scope,
        point.selection,
        None,
        point.line,
    )


def _line_point_matches(point: LineMovementPoint, filters: DashboardFilters) -> bool:
    if filters.events and point.event_id not in filters.events:
        return False
    if filters.sportsbooks and point.sportsbook not in filters.sportsbooks:
        return False
    if filters.markets and point.market not in filters.markets:
        return False
    scope = normalize_scope(point.scope)
    if not filters.include_quarters and is_quarter_scope(scope):
        return False
    if not filters.include_halves and is_half_scope(scope):
        return False
    if filters.scopes and scope not in filters.scopes:
        return False
    return True


def _calibration_matches(point: CalibrationPoint, filters: DashboardFilters) -> bool:
    if filters.markets and point.market not in filters.markets:
        return False
    return True


def _position_matches(position: PortfolioPosition, filters: DashboardFilters) -> bool:
    if filters.events and position.event_id not in filters.events:
        return False
    if filters.sportsbooks and position.sportsbook not in filters.sportsbooks:
        return False
    if filters.markets and position.market not in filters.markets:
        return False
    scope = normalize_scope(position.scope)
    if not filters.include_quarters and is_quarter_scope(scope):
        return False
    if not filters.include_halves and is_half_scope(scope):
        return False
    if filters.scopes and scope not in filters.scopes:
        return False
    return True

