"""Integration-style tests for the betting CLI wiring."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import types

import pytest


def _build_yaml_stub() -> types.ModuleType:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(stream):
        text = stream.read() if hasattr(stream, "read") else str(stream)
        data: dict[str, object] = {}
        for line in str(text).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            raw = value.strip().strip("'\"")
            lowered = raw.lower()
            if lowered in {"true", "false"}:
                data[key] = lowered == "true"
            else:
                try:
                    data[key] = int(raw)
                except ValueError:
                    try:
                        data[key] = float(raw)
                    except ValueError:
                        data[key] = raw
        return data

    yaml_stub.safe_load = _safe_load  # type: ignore[attr-defined]
    return yaml_stub


def _build_pydantic_stub() -> types.ModuleType:
    pydantic_stub = types.ModuleType("pydantic")

    class _StubBaseModel:
        def __init__(self, **values):
            for key, value in values.items():
                setattr(self, key, value)

    def _stub_field(*, default=None, default_factory=None, **kwargs):  # type: ignore[override]
        del kwargs
        if default is not None:
            return default
        if default_factory is not None:
            return default_factory()
        return None

    pydantic_stub.BaseModel = _StubBaseModel  # type: ignore[attr-defined]
    pydantic_stub.Field = _stub_field  # type: ignore[attr-defined]
    return pydantic_stub


@pytest.fixture()
def cli_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "nflreadpy.betting.cli"
    monkeypatch.setitem(sys.modules, "yaml", _build_yaml_stub())
    monkeypatch.setitem(sys.modules, "pydantic", _build_pydantic_stub())

    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("command", ["ingest", "simulate", "scan", "dashboard", "backtest"])
def test_cli_parser_registers_subcommands(cli_module, command: str) -> None:
    parser = cli_module._build_parser()
    args = parser.parse_args([command])
    assert args.command == command
    assert asyncio.iscoroutinefunction(args.handler)


def test_ingest_command_uses_scheduler(
    cli_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class DummyService:
        def __init__(self) -> None:
            self.metrics: dict[str, int] = {}

        async def fetch_and_store(self) -> list[object]:
            events.append("fetched")
            return []

    class DummyScheduler:
        def __init__(self) -> None:
            self.jobs: list[dict[str, object]] = []
            events.append("initialised")

        async def __aenter__(self) -> DummyScheduler:
            events.append("entered")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            events.append("exited")
            await self.shutdown()

        def add_job(
            self,
            action,
            *,
            interval: float,
            jitter: float,
            retries: int,
            retry_backoff: float,
            name: str,
        ) -> object:
            self.jobs.append(
                {
                    "action": action,
                    "interval": interval,
                    "jitter": jitter,
                    "retries": retries,
                    "retry_backoff": retry_backoff,
                    "name": name,
                }
            )
            return object()

        def stop(self) -> None:
            events.append("stopped")

        async def run(self) -> None:
            assert self.jobs
            await self.jobs[0]["action"]()

        async def shutdown(self) -> None:
            events.append("shutdown")

    created: dict[str, DummyScheduler] = {}

    def scheduler_factory() -> DummyScheduler:
        scheduler = DummyScheduler()
        created["scheduler"] = scheduler
        return scheduler

    monkeypatch.setattr(cli_module, "Scheduler", scheduler_factory)
    monkeypatch.setattr(
        cli_module, "install_signal_handlers", lambda callback: events.append("signals")
    )

    context = cli_module.CommandContext(
        service=DummyService(), alert_manager=None, config=object()
    )
    args = argparse.Namespace(
        interval=1.0,
        jitter=0.25,
        retries=2,
        retry_backoff=0.1,
        storage=":memory:",
    )

    asyncio.run(cli_module._cmd_ingest(context, args))

    assert "fetched" in events
    assert "signals" in events
    assert "shutdown" in events
    scheduler = created["scheduler"]
    assert scheduler.jobs and scheduler.jobs[0]["name"] == "odds-ingest"
    assert scheduler.jobs[0]["interval"] == pytest.approx(1.0)
    assert scheduler.jobs[0]["jitter"] == pytest.approx(0.25)
    assert scheduler.jobs[0]["retries"] == 2
    assert scheduler.jobs[0]["retry_backoff"] == pytest.approx(0.1)


def test_portfolio_allocation_prints_extended_summary(
    cli_module, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimise(self, opportunities):
            return [(opp, 1.0) for opp in opportunities]

    monkeypatch.setattr(cli_module, "QuantumPortfolioOptimizer", DummyOptimizer)

    opportunity = cli_module.Opportunity(
        event_id="E1",
        sportsbook="book",
        book_market_group="Game Lines",
        market="moneyline",
        scope="game",
        entity_type="team",
        team_or_player="NE",
        side=None,
        line=None,
        american_odds=+110,
        model_probability=0.55,
        push_probability=0.0,
        implied_probability=0.476,
        expected_value=0.05,
        kelly_fraction=0.1,
        extra={},
    )

    cli_module._portfolio_allocation(
        [opportunity],
        bankroll=100.0,
        portfolio_fraction=1.0,
        correlation_limits={},
        risk_trials=5,
        risk_seed=3,
    )

    out = capsys.readouterr().out
    assert "Bankroll simulation summary:" in out
    assert "Median terminal" in out
    assert "Worst drawdown" in out
    assert "95th percentile drawdown" in out
