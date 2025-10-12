"""Unit tests covering individual alert sink implementations."""
from __future__ import annotations

import json
from email.message import EmailMessage

import pytest

from nflreadpy.betting.alerts import EmailAlertSink, SlackAlertSink, SMSAlertSink


def test_slack_alert_sink_formats_payload() -> None:
    calls: list[tuple[str, dict]] = []

    def transport(url: str, payload: bytes) -> None:
        calls.append((url, json.loads(payload.decode("utf-8"))))

    sink = SlackAlertSink("https://example.test/webhook", transport=transport)
    sink.send("Edge detected", "NE @ +120", metadata={"count": 1})

    assert calls
    url, payload = calls[0]
    assert url == "https://example.test/webhook"
    assert "Edge detected" in payload["text"]
    assert "NE" in payload["text"]
    assert payload["metadata"] == {"count": 1}


def test_email_alert_sink_sets_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[EmailMessage] = []

    def transport(host: str, port: int, message: EmailMessage) -> None:
        captured.append(message)
        assert host == "smtp.test"
        assert port == 2525

    sink = EmailAlertSink(
        sender="alerts@nfl.test",
        recipients=("ops@nfl.test", "betting@nfl.test"),
        host="smtp.test",
        port=2525,
        transport=transport,
    )
    sink.send("Ingestion failure", "All quotes discarded", metadata={"requested": 5})

    assert captured
    message = captured[0]
    assert message["Subject"] == "Ingestion failure"
    assert message["From"] == "alerts@nfl.test"
    assert message["To"] == "ops@nfl.test, betting@nfl.test"
    assert json.loads(message["X-NFLREADPY-Meta"]) == {"requested": 5}
    assert "All quotes discarded" in message.get_content()


def test_sms_alert_sink_invokes_transport() -> None:
    messages: list[tuple[str, str]] = []

    def transport(number: str, body: str) -> None:
        messages.append((number, body))

    sink = SMSAlertSink(("+15550001", "+15550002"), transport=transport)
    sink.send("Line move", "Spread shifted", metadata={"delta": 4})

    assert len(messages) == 2
    for number, body in messages:
        assert number in {"+15550001", "+15550002"}
        assert "Line move" in body
        assert "delta" in body
