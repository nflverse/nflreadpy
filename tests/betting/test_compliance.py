import logging

import pytest

from nflreadpy.betting.analytics import Opportunity, PortfolioManager
from nflreadpy.betting.compliance import (
    ComplianceConfig,
    ComplianceEngine,
    ResponsibleGamingControls,
)


@pytest.fixture
def opportunity_template() -> dict:
    return {
        "event_id": "E1",
        "sportsbook": "book",
        "book_market_group": "Game Lines",
        "market": "spread",
        "scope": "game",
        "entity_type": "team",
        "team_or_player": "NE",
        "side": "fav",
        "line": -3.5,
        "american_odds": -110,
        "model_probability": 0.55,
        "push_probability": 0.05,
        "implied_probability": 0.5238,
        "expected_value": 0.04,
        "kelly_fraction": 0.3,
        "extra": {
            "market_rules": {"push_handling": "push", "includes_overtime": True},
            "jurisdictions": ["nj", "ny"],
        },
    }


def test_compliance_rejects_and_logs(caplog: pytest.LogCaptureFixture, opportunity_template: dict) -> None:
    logger = logging.getLogger("test.audit")
    config = ComplianceConfig(
        allowed_push_handling={"push"},
        require_overtime_included=True,
        jurisdiction_allowlist={"co"},
    )
    engine = ComplianceEngine(config, audit_logger=logger)
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=0.1,
        max_event_exposure=0.5,
        compliance_engine=engine,
        audit_logger=logger,
    )

    caplog.set_level(logging.WARNING, logger="test.audit")
    opportunity = Opportunity(**{
        **opportunity_template,
        "extra": {
            "market_rules": {"push_handling": "lose", "includes_overtime": False},
            "jurisdictions": ["ny"],
        },
    })
    result = manager.allocate(opportunity)
    assert result is None
    assert any(rec.getMessage() == "compliance.violation" for rec in caplog.records)
    assert any(
        rec.getMessage() == "portfolio.rejected" and getattr(rec, "reason", "") == "compliance"
        for rec in caplog.records
    )


def test_session_loss_limit_triggers_cooldown(caplog: pytest.LogCaptureFixture, opportunity_template: dict) -> None:
    logger = logging.getLogger("test.audit.loss")
    controls = ResponsibleGamingControls(session_loss_limit=10.0, cooldown_seconds=30.0)
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=0.2,
        max_event_exposure=1.0,
        responsible_gaming=controls,
        audit_logger=logger,
    )
    caplog.set_level(logging.INFO, logger="test.audit.loss")
    opportunity = Opportunity(**opportunity_template)

    first = manager.allocate(opportunity)
    assert first is not None
    second = manager.allocate(opportunity)
    assert second is None
    assert any(
        rec.getMessage() == "portfolio.rejected" and getattr(rec, "reason", "") == "session_loss_limit"
        for rec in caplog.records
    )
    assert manager._cooldown_until is not None


def test_session_stake_limit_blocks_allocation(caplog: pytest.LogCaptureFixture, opportunity_template: dict) -> None:
    logger = logging.getLogger("test.audit.stake")
    controls = ResponsibleGamingControls(session_stake_limit=5.0)
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=0.2,
        max_event_exposure=1.0,
        responsible_gaming=controls,
        audit_logger=logger,
    )
    caplog.set_level(logging.INFO, logger="test.audit.stake")
    opportunity = Opportunity(**opportunity_template)

    first = manager.allocate(opportunity)
    assert first is not None
    second = manager.allocate(opportunity)
    assert second is None
    assert any(
        rec.getMessage() == "portfolio.rejected" and getattr(rec, "reason", "") == "session_stake_limit"
        for rec in caplog.records
    )
