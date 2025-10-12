"""Compliance and responsible gaming utilities for betting workflows."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

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
    credential_requirements: dict[str, set[str]] = dataclasses.field(
        default_factory=dict
    )
    credentials_available: dict[str, set[str]] = dataclasses.field(
        default_factory=dict
    )
    required_metadata_fields: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        self.allowed_push_handling = {
            str(item).lower() for item in self.allowed_push_handling
        }
        if self.jurisdiction_allowlist is not None:
            self.jurisdiction_allowlist = {
                str(item).lower() for item in self.jurisdiction_allowlist
            }
        self.banned_sportsbooks = {str(item).lower() for item in self.banned_sportsbooks}
        self.credential_requirements = self._normalise_credential_map(
            self.credential_requirements
        )
        self.credentials_available = self._normalise_credential_map(
            self.credentials_available
        )
        self.required_metadata_fields = {
            str(item).strip() for item in self.required_metadata_fields if str(item).strip()
        }

    @staticmethod
    def _normalise_credential_map(
        payload: Mapping[str, Iterable[str]] | MutableMapping[str, Iterable[str]]
        | None,
    ) -> dict[str, set[str]]:
        if not payload:
            return {}
        normalised: dict[str, set[str]] = {}
        for sportsbook, keys in payload.items():
            sportsbook_key = str(sportsbook).strip().lower()
            normalised[sportsbook_key] = {
                str(item).strip().lower() for item in keys if str(item).strip()
            }
        return normalised

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
        requirements = cls._coerce_credential_map(payload.get("credential_requirements"))
        available = cls._coerce_credential_map(payload.get("credentials_available"))
        required_metadata = _to_set("required_metadata_fields") or set()

        return cls(
            allowed_push_handling=allowed_push or cls().allowed_push_handling,
            require_overtime_included=bool(payload.get("require_overtime_included", False)),
            jurisdiction_allowlist=_to_set("jurisdiction_allowlist"),
            banned_sportsbooks=_to_set("banned_sportsbooks") or set(),
            credential_requirements=requirements or {},
            credentials_available=available or {},
            required_metadata_fields=required_metadata,
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
        required = os.getenv(f"{prefix}REQUIRED_CREDENTIALS")
        if required:
            payload["credential_requirements"] = cls._parse_credential_env(required)
        available = os.getenv(f"{prefix}CREDENTIALS_AVAILABLE")
        if available:
            payload["credentials_available"] = cls._parse_credential_env(available)
        required_metadata = os.getenv(f"{prefix}REQUIRED_METADATA_FIELDS")
        if required_metadata:
            payload["required_metadata_fields"] = required_metadata
        return cls.from_mapping(payload)

    @staticmethod
    def _coerce_credential_map(value: object) -> dict[str, set[str]] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return ComplianceConfig._normalise_credential_map(value)
        raise TypeError(
            "credential mappings must be dictionaries of sportsbook -> sequence of keys"
        )

    @staticmethod
    def _parse_credential_env(value: str) -> dict[str, set[str]]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return ComplianceConfig._normalise_credential_map(parsed)
        mapping: dict[str, set[str]] = {}
        entries = [item for item in value.split(";") if item.strip()]
        for entry in entries:
            sportsbook, _, keys = entry.partition(":")
            sportsbook_key = sportsbook.strip().lower()
            if not sportsbook_key:
                continue
            key_items = [
                item.strip().lower() for item in keys.split(",") if item.strip()
            ]
            mapping[sportsbook_key] = set(key_items)
        return mapping


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

        metadata = {}
        extra = getattr(opportunity, "extra", None)
        if isinstance(extra, Mapping):
            metadata = dict(extra)
        compliant, _ = self.evaluate_metadata(
            sportsbook=getattr(opportunity, "sportsbook", ""),
            market=getattr(opportunity, "market", None),
            event_id=getattr(opportunity, "event_id", None),
            metadata=metadata,
            log=True,
        )
        return compliant

    def evaluate_metadata(
        self,
        *,
        sportsbook: str,
        metadata: Mapping[str, Any] | None = None,
        market: str | None = None,
        event_id: str | None = None,
        log: bool = False,
    ) -> tuple[bool, list[str]]:
        """Return a compliance decision and violation reasons for metadata."""

        reasons = self._collect_violations(sportsbook, metadata or {})
        if reasons and log:
            self._log_violation(
                sportsbook=sportsbook,
                market=market,
                event_id=event_id,
                reasons=reasons,
            )
        return (not reasons, reasons)

    def _collect_violations(
        self, sportsbook: str, metadata: Mapping[str, Any]
    ) -> list[str]:
        reasons: list[str] = []
        sportsbook_key = str(sportsbook or "").strip().lower()
        market_rules_raw = metadata.get("market_rules")
        if isinstance(market_rules_raw, Mapping):
            market_rules = {
                str(key): value for key, value in market_rules_raw.items()
            }
        else:
            market_rules = {}

        push = str(market_rules.get("push_handling", "")).strip().lower()
        if self.config.allowed_push_handling:
            allowed_push = {
                item.lower() for item in self.config.allowed_push_handling
            }
            if push:
                if push not in allowed_push:
                    reasons.append(f"push_handling={push}")
            else:
                reasons.append("push_handling=missing")

        includes_ot = market_rules.get("includes_overtime")
        if self.config.require_overtime_included and not bool(includes_ot):
            reasons.append("overtime_not_included")

        if (
            self.config.banned_sportsbooks
            and sportsbook_key in self.config.banned_sportsbooks
        ):
            reasons.append(f"sportsbook_banned={sportsbook_key}")

        if self.config.credential_requirements and sportsbook_key:
            required = self.config.credential_requirements.get(sportsbook_key, set())
            if required:
                provided = set(
                    self.config.credentials_available.get(sportsbook_key, set())
                )
                credentials_meta = metadata.get("credentials")
                if isinstance(credentials_meta, Mapping):
                    provided.update(
                        {
                            str(key).strip().lower()
                            for key, value in credentials_meta.items()
                            if value not in {None, ""}
                        }
                    )
                missing = required.difference(provided)
                if missing:
                    reasons.append(
                        "credentials_missing=" + ",".join(sorted(missing))
                    )

        jurisdictions_raw = metadata.get("jurisdictions")
        if isinstance(jurisdictions_raw, str):
            jurisdictions_iter: Iterable[str] = [
                item.strip() for item in jurisdictions_raw.split(",")
            ]
        elif isinstance(jurisdictions_raw, Iterable):
            jurisdictions_iter = jurisdictions_raw  # type: ignore[assignment]
        else:
            jurisdictions_iter = []
        jurisdictions = list(
            filter(None, (str(item).strip() for item in jurisdictions_iter))
        )
        if self.config.jurisdiction_allowlist is not None:
            allowed = {item.lower() for item in self.config.jurisdiction_allowlist}
            available = {item.lower() for item in jurisdictions}
            if not available:
                reasons.append("jurisdiction=missing")
            elif available.isdisjoint(allowed):
                reasons.append(
                    "jurisdiction_not_permitted=" + ",".join(sorted(available))
                )

        if self.config.required_metadata_fields:
            for field_name in self.config.required_metadata_fields:
                if not metadata.get(field_name):
                    reasons.append(f"metadata_missing={field_name}")

        return reasons

    def _log_violation(
        self,
        *,
        sportsbook: str,
        market: str | None,
        event_id: str | None,
        reasons: Sequence[str],
    ) -> None:
        self._audit_logger.warning(
            "compliance.violation",
            extra={
                "event_id": event_id,
                "market": market,
                "sportsbook": sportsbook,
                "reasons": list(reasons),
            },
        )

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

