"""FastAPI application exposing betting dashboard data."""

from __future__ import annotations

import dataclasses

from ..analytics import Opportunity
from ..dashboard import Dashboard
from ..ingestion import IngestedOdds
from .app import DashboardDataProvider, LineMovementPoint, PortfolioPosition

try:  # pragma: no cover - optional dependency import guard
    from fastapi import Depends, FastAPI
except ModuleNotFoundError:  # pragma: no cover - handled in create_api_app
    Depends = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment]


def create_api_app(provider: DashboardDataProvider) -> "FastAPI":
    """Return a FastAPI app serving dashboard datasets as JSON."""

    if FastAPI is None:  # pragma: no cover - protective runtime guard
        raise RuntimeError(
            "fastapi is required for the betting API. Install with 'uv add fastapi'."
        )

    dashboard = Dashboard()
    app = FastAPI(title="NFL Betting API")

    def _provider() -> DashboardDataProvider:
        return provider

    @app.get("/markets")
    def markets(data: DashboardDataProvider = Depends(_provider)) -> list[dict[str, object]]:
        quotes = list(data.live_markets())
        return [_quote_payload(quote) for quote in quotes]

    @app.get("/opportunities")
    def opportunities(data: DashboardDataProvider = Depends(_provider)) -> list[dict[str, object]]:
        records = list(getattr(data, "opportunities", lambda: [])())
        return [_opportunity_payload(opp) for opp in records]

    @app.get("/line-history")
    def line_history(data: DashboardDataProvider = Depends(_provider)) -> list[dict[str, object]]:
        points = list(data.line_history())
        return [_line_payload(point) for point in points]

    @app.get("/calibration")
    def calibration(data: DashboardDataProvider = Depends(_provider)) -> list[dict[str, object]]:
        points = list(data.calibration())
        return [dataclasses.asdict(point) for point in points]

    @app.get("/portfolio")
    def portfolio(data: DashboardDataProvider = Depends(_provider)) -> list[dict[str, object]]:
        positions = list(data.portfolio())
        return [_portfolio_payload(position) for position in positions]

    @app.get("/filters")
    def filters(data: DashboardDataProvider = Depends(_provider)) -> dict[str, list[str]]:
        odds = list(data.live_markets())
        opportunities = list(getattr(data, "opportunities", lambda: [])())
        return dashboard.available_options(odds, opportunities)

    return app


def _quote_payload(quote: IngestedOdds) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": quote.event_id,
        "sportsbook": quote.sportsbook,
        "book_market_group": quote.book_market_group,
        "market": quote.market,
        "scope": quote.scope,
        "entity_type": quote.entity_type,
        "selection": quote.team_or_player,
        "side": quote.side,
        "line": quote.line,
        "american_odds": quote.american_odds,
        "observed_at": quote.observed_at.isoformat(),
    }
    if quote.extra:
        payload["extra"] = dict(quote.extra)
    return payload


def _opportunity_payload(opportunity: Opportunity) -> dict[str, object]:
    return {
        "event_id": opportunity.event_id,
        "sportsbook": opportunity.sportsbook,
        "market": opportunity.market,
        "scope": opportunity.scope,
        "selection": opportunity.team_or_player,
        "side": opportunity.side,
        "line": opportunity.line,
        "american_odds": opportunity.american_odds,
        "model_probability": opportunity.model_probability,
        "push_probability": opportunity.push_probability,
        "implied_probability": opportunity.implied_probability,
        "expected_value": opportunity.expected_value,
        "kelly_fraction": opportunity.kelly_fraction,
        "extra": dict(opportunity.extra),
    }


def _line_payload(point: LineMovementPoint) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": point.event_id,
        "sportsbook": point.sportsbook,
        "market": point.market,
        "scope": point.scope,
        "selection": point.selection,
        "american_odds": point.american_odds,
        "observed_at": point.observed_at.isoformat(),
    }
    if point.line is not None:
        payload["line"] = point.line
    return payload


def _portfolio_payload(position: PortfolioPosition) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": position.event_id,
        "sportsbook": position.sportsbook,
        "market": position.market,
        "scope": position.scope,
        "selection": position.selection,
        "stake": position.stake,
        "price": position.price,
        "status": position.status,
    }
    if position.expected_value is not None:
        payload["expected_value"] = position.expected_value
    if position.kelly_fraction is not None:
        payload["kelly_fraction"] = position.kelly_fraction
    if position.pnl is not None:
        payload["pnl"] = position.pnl
    return payload
