"""Compliance and responsible gaming utilities for betting workflows."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .analytics import Opportunity


@dataclasses.dataclass(slots=True)
class ComplianceConfig:
    """Configuration governing opportunity eligibility.

    The configuration can be created programmatically, from environment
    variables, or by deserialising YAML/JSON dictionaries.  ``allowed``
    collections are compared in a case-insensitive manner.
    """

    allowed_push_handling: set[str] = dataclasses.field(
        default_factory=lambda: {"push", "refund"}
    )
    require_overtime_included: bool = False
    jurisdiction_allowlist: set[str] | None = None
    banned_sportsbooks: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        self.allowed_push_handling = {
            str(item).lower() for item in self.allowed_push_handling
        }
        if self.jurisdiction_allowlist is not None:
            self.jurisdiction_allowlist = {
                str(item).lower() for item in self.jurisdiction_allowlist
            }
        self.banned_sportsbooks = {str(item).lower() for item in self.banned_sportsbooks}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ComplianceConfig":
        """Construct configuration from a mapping (e.g. parsed YAML)."""

        def _to_set(key: str) -> set[str] | None:
            value = payload.get(key)
            if value is None:
                return None
            if isinstance(value, str):
                items: Iterable[str] = value.split(",")
            elif isinstance(value, Iterable):
                items = value  # type: ignore[assignment]
            else:  # pragma: no cover - defensive
                raise TypeError(f"{key} must be a string or iterable of strings")
            return {str(item).strip().lower() for item in items if str(item).strip()}

        allowed_push = _to_set("allowed_push_handling")
        return cls(
            allowed_push_handling=allowed_push or cls().allowed_push_handling,
            require_overtime_included=bool(payload.get("require_overtime_included", False)),
            jurisdiction_allowlist=_to_set("jurisdiction_allowlist"),
            banned_sportsbooks=_to_set("banned_sportsbooks") or set(),
        )

    @classmethod
    def from_env(cls, prefix: str = "NFLREADPY_COMPLIANCE_") -> "ComplianceConfig":
        """Construct configuration using environment variables.

        Supported variables:

        ``<prefix>ALLOWED_PUSH_HANDLING``
            Comma separated list of accepted push handling treatments.
        ``<prefix>REQUIRE_OVERTIME_INCLUDED``
            ``1``/``0`` toggle for requiring overtime inclusion metadata.
        ``<prefix>JURISDICTION_ALLOWLIST``
            Comma separated list of jurisdictions the bettor is licensed for.
        ``<prefix>BANNED_SPORTSBOOKS``
            Comma separated list of sportsbooks that should never be wagered.
        """

        payload: dict[str, object] = {}
        allowed = os.getenv(f"{prefix}ALLOWED_PUSH_HANDLING")
        if allowed:
            payload["allowed_push_handling"] = allowed
        overtime = os.getenv(f"{prefix}REQUIRE_OVERTIME_INCLUDED")
        if overtime is not None:
            payload["require_overtime_included"] = overtime not in {"0", "false", "False"}
        jurisdictions = os.getenv(f"{prefix}JURISDICTION_ALLOWLIST")
        if jurisdictions:
            payload["jurisdiction_allowlist"] = jurisdictions
        banned = os.getenv(f"{prefix}BANNED_SPORTSBOOKS")
        if banned:
            payload["banned_sportsbooks"] = banned
        return cls.from_mapping(payload)


@dataclasses.dataclass(slots=True)
class ResponsibleGamingControls:
    """Session-level controls limiting bankroll depletion."""

    session_loss_limit: float | None = None
    session_stake_limit: float | None = None
    cooldown_seconds: float | None = None

    @classmethod
    def from_env(cls, prefix: str = "NFLREADPY_RESPONSIBLE_") -> "ResponsibleGamingControls":
        """Create controls from environment variables."""

        def _parse_float(value: str | None) -> float | None:
            if not value:
                return None
            try:
                return float(value)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Could not parse float from {value}") from exc

        return cls(
            session_loss_limit=_parse_float(os.getenv(f"{prefix}SESSION_LOSS_LIMIT")),
            session_stake_limit=_parse_float(os.getenv(f"{prefix}SESSION_STAKE_LIMIT")),
            cooldown_seconds=_parse_float(os.getenv(f"{prefix}COOLDOWN_SECONDS")),
        )


class ComplianceEngine:
    """Evaluate betting opportunities against compliance policies."""

    def __init__(
        self,
        config: ComplianceConfig,
        audit_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._audit_logger = audit_logger or logging.getLogger("nflreadpy.betting.audit")

    def validate(self, opportunity: "Opportunity") -> bool:
        """Return ``True`` when the opportunity satisfies compliance checks."""

        reasons = []
        metadata = dict(getattr(opportunity, "extra", {}) or {})
        market_rules_raw = metadata.get("market_rules", {})
        if isinstance(market_rules_raw, Mapping):
            market_rules = {str(key): value for key, value in market_rules_raw.items()}
        else:  # pragma: no cover - defensive guard
            market_rules = {}

        push = str(market_rules.get("push_handling", "")).lower()
        if self.config.allowed_push_handling and push:
            if push not in {item.lower() for item in self.config.allowed_push_handling}:
                reasons.append(f"push_handling={push}")
        elif self.config.allowed_push_handling:
            reasons.append("push_handling=missing")

        includes_ot = market_rules.get("includes_overtime")
        if self.config.require_overtime_included and not bool(includes_ot):
            reasons.append("overtime_not_included")

        if self.config.banned_sportsbooks:
            sportsbook = getattr(opportunity, "sportsbook", "").lower()
            if sportsbook in self.config.banned_sportsbooks:
                reasons.append(f"sportsbook_banned={sportsbook}")

        jurisdictions_raw = metadata.get("jurisdictions")
        if isinstance(jurisdictions_raw, str):
            jurisdictions_iter: Iterable[str] = [
                item.strip() for item in jurisdictions_raw.split(",")
            ]
        elif isinstance(jurisdictions_raw, Iterable):
            jurisdictions_iter = jurisdictions_raw  # type: ignore[assignment]
        else:
            jurisdictions_iter = []
        jurisdictions = list(filter(None, (str(item).strip() for item in jurisdictions_iter)))
        if self.config.jurisdiction_allowlist is not None:
            allowed = {item.lower() for item in self.config.jurisdiction_allowlist}
            available = {item.lower() for item in jurisdictions}
            if not available:
                reasons.append("jurisdiction=missing")
            elif available.isdisjoint(allowed):
                reasons.append(
                    "jurisdiction_not_permitted=" + ",".join(sorted(available))
                )

        if reasons:
            self._audit_logger.warning(
                "compliance.violation",
                extra={
                    "event_id": getattr(opportunity, "event_id", None),
                    "market": getattr(opportunity, "market", None),
                    "sportsbook": getattr(opportunity, "sportsbook", None),
                    "reasons": reasons,
                },
            )
            return False
        return True

    def filter(self, opportunities: Sequence["Opportunity"]) -> list["Opportunity"]:
        """Return only compliant opportunities."""

        return [
            opportunity
            for opportunity in opportunities
            if self.validate(opportunity)
        ]


__all__ = [
    "ComplianceConfig",
    "ComplianceEngine",
    "ResponsibleGamingControls",
]

