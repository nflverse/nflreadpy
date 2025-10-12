from __future__ import annotations

from pathlib import Path

import pytest

from nflreadpy.betting.alerts import AlertManager
from nflreadpy.betting.scheduler import BacktestOrchestrator, BacktestScheduleConfig


SAMPLE_SNAPSHOTS = (
    Path(__file__).resolve().parent / "data" / "historical_snapshots.csv"
).read_text(encoding="utf-8")


class RecordingSink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict | None]] = []

    def send(self, subject: str, body: str, *, metadata=None) -> None:
        self.messages.append((subject, body, metadata))


@pytest.mark.asyncio
async def test_weekly_backtest_scheduler_persists_reports_and_alerts(tmp_path: Path) -> None:
    season_dir = tmp_path / "snapshots" / "2024"
    season_dir.mkdir(parents=True)
    (season_dir / "week_01.csv").write_text(SAMPLE_SNAPSHOTS, encoding="utf-8")

    sink = RecordingSink()
    manager = AlertManager([sink])
    config = BacktestScheduleConfig(
        snapshot_root=tmp_path / "snapshots",
        output_root=tmp_path / "reports",
        cadence="weekly",
        alert_manager=manager,
    )
    orchestrator = BacktestOrchestrator(config)

    await orchestrator.run_once()

    week_one_dir = config.output_root / "2024" / "week01"
    assert (week_one_dir / "reliability_diagram.csv").exists()
    assert len(sink.messages) == 1
    first_subject, _, first_meta = sink.messages[0]
    assert "completed" in first_subject.lower()
    assert first_meta and first_meta.get("week") == 1

    # Running again without new data should not duplicate alerts.
    await orchestrator.run_once()
    assert len(sink.messages) == 1

    # Introduce a new week and ensure it is processed on the next interval.
    (season_dir / "week02.csv").write_text(SAMPLE_SNAPSHOTS, encoding="utf-8")
    await orchestrator.run_once()

    week_two_dir = config.output_root / "2024" / "week02"
    assert (week_two_dir / "closing_line_report.csv").exists()
    assert len(sink.messages) == 2
    _, _, second_meta = sink.messages[-1]
    assert second_meta and second_meta.get("week") == 2


@pytest.mark.asyncio
async def test_seasonal_backtest_scheduler_persists_reports(tmp_path: Path) -> None:
    season_dir = tmp_path / "snapshots" / "2023"
    season_dir.mkdir(parents=True)
    (season_dir / "week1.csv").write_text(SAMPLE_SNAPSHOTS, encoding="utf-8")

    sink = RecordingSink()
    manager = AlertManager([sink])
    config = BacktestScheduleConfig(
        snapshot_root=tmp_path / "snapshots",
        output_root=tmp_path / "reports",
        cadence="seasonal",
        alert_manager=manager,
    )
    orchestrator = BacktestOrchestrator(config)

    await orchestrator.run_once()

    season_dir_out = config.output_root / "2023" / "season"
    assert (season_dir_out / "reliability_diagram.csv").exists()
    assert sink.messages
    subject, _, metadata = sink.messages[0]
    assert "completed" in subject.lower()
    assert metadata and metadata.get("season") == 2023
    assert metadata.get("week") is None or "week" not in metadata


@pytest.mark.asyncio
async def test_backtest_scheduler_emits_failure_alert(tmp_path: Path) -> None:
    season_dir = tmp_path / "snapshots" / "2025"
    season_dir.mkdir(parents=True)
    # Missing required columns will cause the backtest to raise a ValueError.
    (season_dir / "week03.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

    sink = RecordingSink()
    manager = AlertManager([sink])
    config = BacktestScheduleConfig(
        snapshot_root=tmp_path / "snapshots",
        output_root=tmp_path / "reports",
        cadence="weekly",
        alert_manager=manager,
    )
    orchestrator = BacktestOrchestrator(config)

    await orchestrator.run_once()

    assert sink.messages
    subject, body, metadata = sink.messages[0]
    assert "failed" in subject.lower()
    assert "2025" in body
    assert metadata and metadata.get("week") == 3

    # The failing task should be retried on subsequent runs.
    await orchestrator.run_once()
    assert len(sink.messages) == 2
