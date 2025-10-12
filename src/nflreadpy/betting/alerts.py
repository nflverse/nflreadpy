"""Alert routing utilities for sportsbook opportunity monitoring."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import signal
from collections.abc import Callable, Mapping, Sequence
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol

try:  # pragma: no cover - Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Py<3.11 fallback
    import tomli as tomllib  # type: ignore


logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    """Protocol describing a sink that can emit alert messages."""

    def send(self, subject: str, body: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Send a formatted alert message."""


def _default_slack_transport(url: str, payload: bytes) -> None:
    from urllib import request

    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as response:  # pragma: no branch - tiny wrapper
        response.read()


def _default_email_transport(host: str, port: int, message: EmailMessage) -> None:
    import smtplib

    with smtplib.SMTP(host, port, timeout=10) as client:  # pragma: no branch - tiny wrapper
        client.send_message(message)


def _default_sms_transport(number: str, body: str) -> None:
    logger.info("SMS alert to %s: %s", number, body)


@dataclasses.dataclass(slots=True)
class SlackAlertSink:
    """Post alerts to a Slack webhook."""

    webhook_url: str
    transport: Callable[[str, bytes], None] = _default_slack_transport

    def send(
        self, subject: str, body: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        payload = {"text": f"*{subject}*\n{body}"}
        if metadata:
            payload["metadata"] = dict(metadata)
        try:
            self.transport(self.webhook_url, json.dumps(payload).encode("utf-8"))
        except Exception:  # pragma: no cover - logging side effect
            logger.exception("Failed to send Slack alert")


@dataclasses.dataclass(slots=True)
class EmailAlertSink:
    """Send alerts via SMTP email."""

    sender: str
    recipients: Sequence[str]
    host: str = "localhost"
    port: int = 25
    transport: Callable[[str, int, EmailMessage], None] = _default_email_transport

    def send(
        self, subject: str, body: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        if not self.recipients:
            return
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        if metadata:
            message["X-NFLREADPY-Meta"] = json.dumps(dict(metadata), sort_keys=True)
        message.set_content(body)
        try:
            self.transport(self.host, self.port, message)
        except Exception:  # pragma: no cover - logging side effect
            logger.exception("Failed to send email alert")


@dataclasses.dataclass(slots=True)
class SMSAlertSink:
    """Deliver alerts via SMS provider callback."""

    numbers: Sequence[str]
    transport: Callable[[str, str], None] = _default_sms_transport

    def send(
        self, subject: str, body: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        if not self.numbers:
            return
        message_body = f"{subject}: {body}" if subject else body
        if metadata:
            meta = ", ".join(f"{key}={value}" for key, value in metadata.items())
            message_body = f"{message_body} ({meta})"
        for number in self.numbers:
            try:
                self.transport(number, message_body)
            except Exception:  # pragma: no cover - logging side effect
                logger.exception("Failed to send SMS alert")


@dataclasses.dataclass(slots=True)
class AlertConfiguration:
    """Configuration values for alert routing."""

    enabled: bool = False
    slack_webhook: str | None = None
    email_sender: str | None = None
    email_recipients: tuple[str, ...] = ()
    email_host: str = "localhost"
    email_port: int = 25
    sms_numbers: tuple[str, ...] = ()
    jitter_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AlertConfiguration":
        return cls(
            enabled=bool(data.get("enabled", False)),
            slack_webhook=data.get("slack_webhook"),
            email_sender=data.get("email_sender"),
            email_recipients=tuple(data.get("email_recipients", []) or ()),
            email_host=data.get("email_host", "localhost"),
            email_port=int(data.get("email_port", 25)),
            sms_numbers=tuple(data.get("sms_numbers", []) or ()),
            jitter_seconds=float(data.get("jitter_seconds", 0.0)),
        )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except Exception:  # pragma: no cover - optional dependency missing
        data: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if value.lower() in {"true", "false"}:
                    data[key] = value.lower() == "true"
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value
        return data
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_alert_config(path: str | os.PathLike[str] | None = None) -> AlertConfiguration:
    """Load alert configuration from pyproject.toml or a YAML file."""

    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            data = _load_yaml(config_path)
        else:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        for key in ("tool", "nflreadpy", "alerts"):
            if isinstance(data, Mapping) and key in data:
                data = data[key]
            else:
                break
        if not isinstance(data, Mapping):
            data = {}
        return AlertConfiguration.from_mapping(data)

    project_pyproject = Path.cwd() / "pyproject.toml"
    if project_pyproject.exists():
        with project_pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        section = (
            data.get("tool", {})
            .get("nflreadpy", {})
            .get("alerts", {})
        )
        if isinstance(section, Mapping):
            return AlertConfiguration.from_mapping(section)
    return AlertConfiguration()


@dataclasses.dataclass(slots=True)
class AlertManager:
    """Dispatch alert notifications to configured sinks."""

    sinks: Sequence[AlertSink]
    jitter_seconds: float = 0.0

    def send(
        self, subject: str, body: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        if not self.sinks:
            return
        delay = 0.0
        if self.jitter_seconds:
            delay = random.uniform(0.0, self.jitter_seconds)
        if delay:
            import threading

            timer = threading.Timer(delay, self._dispatch, args=(subject, body, metadata))
            timer.daemon = True
            timer.start()
            return
        self._dispatch(subject, body, metadata)

    def _dispatch(
        self, subject: str, body: str, metadata: Mapping[str, Any] | None
    ) -> None:
        for sink in self.sinks:
            try:
                sink.send(subject, body, metadata=metadata)
            except Exception:  # pragma: no cover - sink specific
                logger.exception("Alert sink %s raised", sink)

    def notify_edges(self, opportunities: Sequence[Mapping[str, Any]]) -> None:
        if not opportunities:
            return
        top = min(3, len(opportunities))
        lines = []
        for opp in opportunities[:top]:
            selection = opp.get("team_or_player")
            odds = opp.get("american_odds")
            ev = opp.get("expected_value")
            lines.append(f"{selection} @ {odds} (EV {ev:.2%})")
        body = "\n".join(lines)
        metadata = {"count": len(opportunities)}
        self.send("Edges detected", body, metadata=metadata)

    def notify_line_movement(self, movements: Sequence[Mapping[str, Any]], threshold: int) -> None:
        notable = [movement for movement in movements if abs(int(movement.get("delta", 0))) >= threshold]
        if not notable:
            return
        lines = []
        for movement in notable[:3]:
            key = movement.get("key")
            if isinstance(key, Sequence):
                descriptor = " ".join(str(part) for part in key[:3])
            else:
                descriptor = str(key)
            delta = movement.get("delta")
            lines.append(f"{descriptor} moved {delta:+d}")
        body = "\n".join(lines)
        metadata = {"count": len(notable)}
        self.send("Line movement detected", body, metadata=metadata)

    def notify_ingestion_health(self, metrics: Mapping[str, Any]) -> None:
        requested = int(metrics.get("requested", 0) or 0)
        persisted = int(metrics.get("persisted", 0) or 0)
        discarded_raw = metrics.get("discarded", {})
        discarded_total = 0
        discarded_breakdown: dict[str, int] = {}
        if isinstance(discarded_raw, Mapping):
            for key, value in discarded_raw.items():
                try:
                    count = int(value)
                except Exception:
                    continue
                if count <= 0:
                    continue
                discarded_total += count
                discarded_breakdown[str(key)] = count
        if requested == 0 and persisted == 0 and discarded_total == 0:
            return
        if requested == 0:
            subject = "Ingestion idle"
            body = "No quotes were requested from sportsbooks."
        elif persisted == 0:
            subject = "Ingestion failure"
            body = f"All {requested} requested quotes were discarded."
        elif discarded_total:
            subject = "Ingestion degradation"
            body = f"{discarded_total} of {requested} quotes were discarded."
        else:
            return
        metadata: dict[str, Any] = {
            "requested": requested,
            "persisted": persisted,
        }
        if discarded_breakdown:
            metadata["discarded"] = discarded_breakdown
        self.send(subject, body, metadata=metadata)


_cached_manager: AlertManager | None | bool = False


def get_alert_manager(path: str | os.PathLike[str] | None = None) -> AlertManager | None:
    """Return an alert manager built from configuration, caching the result."""

    global _cached_manager
    if path is None and isinstance(_cached_manager, AlertManager):
        return _cached_manager
    if path is None and _cached_manager is True:
        return None
    config = load_alert_config(path)
    if not config.enabled:
        if path is None:
            _cached_manager = True
        return None
    sinks: list[AlertSink] = []
    if config.slack_webhook:
        sinks.append(SlackAlertSink(config.slack_webhook))
    if config.email_sender:
        sinks.append(
            EmailAlertSink(
                sender=config.email_sender,
                recipients=config.email_recipients,
                host=config.email_host,
                port=config.email_port,
            )
        )
    if config.sms_numbers:
        sinks.append(SMSAlertSink(config.sms_numbers))
    manager = AlertManager(sinks=sinks, jitter_seconds=config.jitter_seconds)
    if path is None:
        _cached_manager = manager
    return manager


class GracefulExit(RuntimeError):
    """Raised when the alert scheduler receives a termination signal."""


def install_signal_handlers(stop_callback: Callable[[], None]) -> None:
    """Install POSIX signal handlers that trigger ``stop_callback``."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no running loop
        loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_callback)
        except NotImplementedError:  # pragma: no cover - Windows
            signal.signal(sig, lambda *_: stop_callback())


__all__ = [
    "AlertConfiguration",
    "AlertManager",
    "AlertSink",
    "EmailAlertSink",
    "GracefulExit",
    "SMSAlertSink",
    "SlackAlertSink",
    "get_alert_manager",
    "install_signal_handlers",
    "load_alert_config",
]

