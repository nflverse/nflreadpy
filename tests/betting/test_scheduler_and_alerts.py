import asyncio
import datetime as dt
from pathlib import Path

from nflreadpy.betting.alerts import (
    AlertManager,
    get_alert_manager,
    load_alert_config,
)
from nflreadpy.betting.analytics import EdgeDetector, LineMovementAnalyzer
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.models import GameSimulationConfig, MonteCarloEngine, TeamRating
from nflreadpy.betting.scheduler import Scheduler
from nflreadpy.betting.scrapers.base import OddsQuote


class RecordingSink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict | None]] = []

    def send(self, subject: str, body: str, *, metadata=None) -> None:
        self.messages.append((subject, body, metadata))


def _build_quotes(now: dt.datetime) -> list[OddsQuote]:
    return [
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="test",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side=None,
            line=None,
            american_odds=-120,
            observed_at=now,
            extra={},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="test",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NYJ",
            side=None,
            line=None,
            american_odds=+110,
            observed_at=now,
            extra={},
        ),
    ]


def test_edge_detector_triggers_alerts() -> None:
    now = dt.datetime(2024, 9, 1, 12, tzinfo=dt.timezone.utc)
    quotes = _build_quotes(now)
    ratings = {
        "NE": TeamRating(team="NE", offensive_rating=1.5, defensive_rating=0.6),
        "NYJ": TeamRating(team="NYJ", offensive_rating=-0.3, defensive_rating=0.4),
    }
    engine = MonteCarloEngine(ratings, GameSimulationConfig(iterations=1_000, seed=21))
    simulation = engine.simulate_game("2024-NE-NYJ", "NE", "NYJ")
    sink = RecordingSink()
    manager = AlertManager([sink])
    detector = EdgeDetector(value_threshold=0.0, alert_manager=manager)
    opportunities = detector.detect(quotes, [simulation])
    assert opportunities
    assert sink.messages
    subject, body, metadata = sink.messages[0]
    assert "Edges" in subject
    assert "NE" in body or "NYJ" in body
    assert metadata and metadata["count"] == len(opportunities)


def test_line_movement_alerts_respect_threshold() -> None:
    now = dt.datetime(2024, 9, 1, 12, tzinfo=dt.timezone.utc)
    earlier = now - dt.timedelta(minutes=10)
    history = [
        IngestedOdds(
            event_id="2024-NE-NYJ",
            sportsbook="test",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side=None,
            line=None,
            american_odds=-110,
            observed_at=earlier,
            extra={},
        ),
        IngestedOdds(
            event_id="2024-NE-NYJ",
            sportsbook="test",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side=None,
            line=None,
            american_odds=-140,
            observed_at=now,
            extra={},
        ),
    ]
    sink = RecordingSink()
    manager = AlertManager([sink])
    analyzer = LineMovementAnalyzer(history, alert_manager=manager, alert_threshold=15)
    movements = analyzer.summarise()
    assert movements
    assert sink.messages
    subject, body, metadata = sink.messages[0]
    assert "Line movement" in subject
    assert "2024-NE-NYJ" in body
    assert metadata and metadata["count"] >= 1


def test_scheduler_retries_and_graceful_shutdown() -> None:
    async def runner() -> None:
        scheduler = Scheduler()
        attempts = 0

        async def failing_job() -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise RuntimeError("boom")

        scheduler.add_job(failing_job, interval=0.01, retries=1, retry_backoff=0.01)
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        scheduler.stop()
        await task
        assert attempts >= 2

    asyncio.run(runner())


def test_load_alert_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "alerts.yaml"
    config_file.write_text(
        "enabled: true\nslack_webhook: https://example.test/webhook\n", encoding="utf-8"
    )
    config = load_alert_config(config_file)
    assert config.enabled is True
    assert config.slack_webhook == "https://example.test/webhook"


def test_get_alert_manager_respects_global_cache(monkeypatch) -> None:
    monkeypatch.setattr("nflreadpy.betting.alerts._cached_manager", False, raising=False)

    class StubConfig:
        enabled = True
        slack_webhook = None
        email_sender = None
        email_recipients: tuple[str, ...] = ()
        email_host = "localhost"
        email_port = 25
        sms_numbers: tuple[str, ...] = ()
        jitter_seconds = 0.0

    monkeypatch.setattr(
        "nflreadpy.betting.alerts.load_alert_config",
        lambda path=None: StubConfig(),
    )
    manager1 = get_alert_manager(None)
    manager2 = get_alert_manager(None)
    assert manager1 is manager2
