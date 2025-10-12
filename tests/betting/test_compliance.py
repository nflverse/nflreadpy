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
    rejection_records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "portfolio.rejected"
        and getattr(rec, "reason", "") == "compliance"
    ]
    assert rejection_records
    assert any(getattr(rec, "reasons", []) for rec in rejection_records)


def test_compliance_rejects_missing_credentials(
    caplog: pytest.LogCaptureFixture, opportunity_template: dict
) -> None:
    logger = logging.getLogger("test.audit.creds")
    config = ComplianceConfig(
        allowed_push_handling={"push"},
        credential_requirements={"book": {"session_token"}},
    )
    engine = ComplianceEngine(config, audit_logger=logger)
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=0.1,
        max_event_exposure=0.5,
        compliance_engine=engine,
        audit_logger=logger,
    )

    caplog.set_level(logging.WARNING, logger="test.audit.creds")
    opportunity = Opportunity(**opportunity_template)
    result = manager.allocate(opportunity)
    assert result is None
    violations = [
        rec for rec in caplog.records if rec.getMessage() == "compliance.violation"
    ]
    assert violations
    assert any(
        "credentials_missing" in ",".join(getattr(rec, "reasons", []))
        for rec in violations
    )
    portfolio_rejections = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "portfolio.rejected"
    ]
    assert any(
        "credentials_missing" in ",".join(getattr(rec, "reasons", []))
        for rec in portfolio_rejections
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


def test_config_from_env_parses_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "NFLREADPY_COMPLIANCE_REQUIRED_CREDENTIALS",
        '{"book": ["session_token", "account_id"]}',
    )
    monkeypatch.setenv(
        "NFLREADPY_COMPLIANCE_CREDENTIALS_AVAILABLE",
        "book:session_token",
    )
    monkeypatch.setenv(
        "NFLREADPY_COMPLIANCE_REQUIRED_METADATA_FIELDS",
        "jurisdictions,market_rules",
    )
    config = ComplianceConfig.from_env()
    assert config.credential_requirements["book"] == {"session_token", "account_id"}
    assert config.credentials_available["book"] == {"session_token"}
    assert config.required_metadata_fields == {"jurisdictions", "market_rules"}


def test_config_from_mapping_handles_yaml_payload() -> None:
    payload = {
        "allowed_push_handling": ["Push", "Refund"],
        "jurisdiction_allowlist": ["NJ", "NY"],
        "banned_sportsbooks": ["OffshoreBook"],
        "credential_requirements": {"FanDuel": ["session_token"]},
        "credentials_available": {"FanDuel": ["Session_Token"]},
        "required_metadata_fields": ["Jurisdictions", "Market_Rules"],
    }
    config = ComplianceConfig.from_mapping(payload)
    assert config.allowed_push_handling == {"push", "refund"}
    assert config.jurisdiction_allowlist == {"nj", "ny"}
    assert config.banned_sportsbooks == {"offshorebook"}
    assert config.credential_requirements["fanduel"] == {"session_token"}
    assert config.credentials_available["fanduel"] == {"session_token"}
    assert config.required_metadata_fields == {"Jurisdictions", "Market_Rules"}


def test_evaluate_metadata_returns_reasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.audit.metadata")
    config = ComplianceConfig(
        jurisdiction_allowlist={"co"}, required_metadata_fields={"jurisdictions"}
    )
    engine = ComplianceEngine(config, audit_logger=logger)

    caplog.set_level(logging.WARNING, logger="test.audit.metadata")
    compliant, reasons = engine.evaluate_metadata(
        sportsbook="book",
        market="spread",
        event_id="E1",
        metadata={"jurisdictions": ["ny"]},
        log=True,
    )
    assert not compliant
    assert reasons
    assert any(
        rec.getMessage() == "compliance.violation" for rec in caplog.records
    )


def test_required_metadata_fields_trigger_violation(
    caplog: pytest.LogCaptureFixture, opportunity_template: dict
) -> None:
    logger = logging.getLogger("test.audit.metadata_required")
    config = ComplianceConfig(required_metadata_fields={"jurisdictions"})
    engine = ComplianceEngine(config, audit_logger=logger)
    manager = PortfolioManager(
        bankroll=100.0,
        max_risk_per_bet=0.1,
        max_event_exposure=0.5,
        compliance_engine=engine,
        audit_logger=logger,
    )

    caplog.set_level(logging.WARNING, logger="test.audit.metadata_required")
    opportunity = Opportunity(**{**opportunity_template, "extra": {}})
    result = manager.allocate(opportunity)
    assert result is None
    violations = [
        rec for rec in caplog.records if rec.getMessage() == "compliance.violation"
    ]
    assert violations
    assert any(
        "metadata_missing=jurisdictions" in ",".join(getattr(rec, "reasons", []))
        for rec in violations
    )
