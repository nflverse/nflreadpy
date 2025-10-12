"""Asynchronous scheduler with jitter, retries, and graceful shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import random
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .alerts import AlertManager
from .backtesting import (
    BacktestArtifacts,
    BacktestMetrics,
    SportsbookRules,
    load_historical_snapshots,
    persist_backtest_reports,
    run_backtest,
)

logger = logging.getLogger(__name__)


AsyncCallable = Callable[[], Awaitable[Any]]


@dataclasses.dataclass(slots=True)
class ScheduledJob:
    """Representation of a coroutine executed at an interval."""

    name: str | None
    action: AsyncCallable
    interval: float
    jitter: float = 0.0
    retries: int = 0
    retry_backoff: float = 2.0

    async def run(self, stop_event: asyncio.Event) -> None:
        """Execute ``action`` until ``stop_event`` is set."""

        attempt = 0
        while not stop_event.is_set():
            try:
                await self.action()
                attempt = 0
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
                raise
            except Exception:  # pragma: no cover - logging side effects
                attempt += 1
                logger.exception("Scheduled job %s failed", self.name)
                if attempt <= self.retries:
                    backoff = max(0.0, self.retry_backoff) * attempt
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                        return
                    except asyncio.TimeoutError:
                        continue
                attempt = 0
            delay = max(0.0, self.interval)
            if delay == 0:
                await asyncio.sleep(0)
                continue
            if self.jitter:
                delay = max(0.0, delay + random.uniform(-self.jitter, self.jitter))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                continue


class Scheduler:
    """Manage a collection of scheduled asynchronous jobs."""

    def __init__(self) -> None:
        self._jobs: list[ScheduledJob] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()

    async def __aenter__(self) -> "Scheduler":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        del exc_type, exc, traceback
        await self.shutdown()

    @property
    def jobs(self) -> Sequence[ScheduledJob]:
        """Return a snapshot of registered jobs."""

        return tuple(self._jobs)

    def add_job(
        self,
        action: AsyncCallable,
        *,
        interval: float,
        jitter: float = 0.0,
        retries: int = 0,
        retry_backoff: float = 2.0,
        name: str | None = None,
    ) -> ScheduledJob:
        job_name = name or getattr(action, "__name__", "scheduled-job")
        job = ScheduledJob(
            name=job_name,
            action=action,
            interval=interval,
            jitter=jitter,
            retries=retries,
            retry_backoff=retry_backoff,
        )
        self._jobs.append(job)
        return job

    def stop(self) -> None:
        """Signal all jobs to cease execution."""

        self._stop_event.set()

    async def run(self) -> None:
        """Run until ``stop`` is called or all jobs finish."""

        if not self._jobs:
            return
        loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._tasks = [
            loop.create_task(job.run(self._stop_event), name=job.name)
            for job in self._jobs
        ]
        stop_task = loop.create_task(self._stop_event.wait(), name="scheduler-stop")
        try:
            await asyncio.wait(
                [stop_task, *self._tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop_task.cancel()
            with contextlib.suppress(Exception):
                await stop_task
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cancel running jobs and wait for them to exit."""

        if not self._tasks:
            return
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


@dataclasses.dataclass(slots=True)
class BacktestTask:
    """Description of a backtest batch discovered on disk."""

    season: int
    week: int | None
    path: Path


@dataclasses.dataclass(slots=True)
class BacktestScheduleConfig:
    """Configuration describing how scheduled backtests should run."""

    snapshot_root: Path
    output_root: Path
    cadence: Literal["weekly", "seasonal"]
    start_season: int | None = None
    end_season: int | None = None
    start_week: int | None = None
    end_week: int | None = None
    sportsbook_rules: Mapping[str, SportsbookRules] | None = None
    alert_manager: AlertManager | None = None
    reliability_bins: int = 10


class BacktestOrchestrator:
    """Discover stored odds snapshots and persist recurring backtest reports."""

    def __init__(self, config: BacktestScheduleConfig) -> None:
        self._config = config
        self._processed: set[tuple[int, int | None]] = set()

    @property
    def cadence(self) -> Literal["weekly", "seasonal"]:
        return self._config.cadence

    async def run_once(self) -> None:
        """Process newly discovered snapshot batches once."""

        tasks = list(self._discover_tasks())
        if not tasks:
            return
        for task in tasks:
            await self._run_task(task)

    async def _run_task(self, task: BacktestTask) -> None:
        logger.debug(
            "Running backtest for season %s week %s from %s",
            task.season,
            task.week,
            task.path,
        )
        try:
            metrics = await asyncio.to_thread(
                self._execute_backtest,
                task,
            )
            artefacts = await asyncio.to_thread(
                self._persist_reports,
                task,
                metrics,
            )
        except Exception as exc:  # pragma: no cover - logged path
            logger.exception(
                "Backtest failed for season %s week %s", task.season, task.week,
            )
            self._notify_failure(task, exc)
            return

        self._processed.add((task.season, task.week))
        self._notify_success(task, metrics, artefacts)

    def _notify_success(
        self,
        task: BacktestTask,
        metrics: BacktestMetrics,
        artefacts: BacktestArtifacts,
    ) -> None:
        manager = self._config.alert_manager
        if not manager:
            return
        metadata: dict[str, Any] = {
            "season": task.season,
            "pnl": round(metrics.total_pnl, 4),
            "settlements": len(metrics.settlements),
        }
        if task.week is not None:
            metadata["week"] = task.week
        body_lines = [
            f"Reliability: {artefacts.reliability_path}",
            f"Closing lines: {artefacts.closing_line_path}",
        ]
        manager.send(
            "Backtest completed",
            "\n".join(body_lines),
            metadata=metadata,
        )

    def _notify_failure(self, task: BacktestTask, exc: Exception) -> None:
        manager = self._config.alert_manager
        if not manager:
            return
        metadata: dict[str, Any] = {"season": task.season}
        descriptor = "season" if task.week is None else f"week {task.week}"
        if task.week is not None:
            metadata["week"] = task.week
        manager.send(
            "Backtest failed",
            f"Backtest for season {task.season} {descriptor} failed: {exc}",
            metadata=metadata,
        )

    def _execute_backtest(self, task: BacktestTask) -> BacktestMetrics:
        snapshots = load_historical_snapshots(task.path)
        return run_backtest(
            snapshots,
            sportsbook_rules=self._config.sportsbook_rules,
        )

    def _persist_reports(
        self,
        task: BacktestTask,
        metrics: BacktestMetrics,
    ) -> BacktestArtifacts:
        output = self._config.output_root / str(task.season)
        if task.week is None:
            output = output / "season"
        else:
            output = output / f"week{task.week:02d}"
        return persist_backtest_reports(
            metrics,
            output,
            bins=self._config.reliability_bins,
        )

    def _discover_tasks(self) -> Iterable[BacktestTask]:
        config = self._config
        root = config.snapshot_root
        if not root.exists():
            logger.debug("Snapshot root %s does not exist", root)
            return []
        if config.cadence == "weekly":
            yield from self._discover_weekly()
        else:
            yield from self._discover_seasonal()

    def _discover_weekly(self) -> Iterable[BacktestTask]:
        config = self._config
        for season_path in sorted(config.snapshot_root.iterdir()):
            if not season_path.is_dir():
                continue
            season = _coerce_int(season_path.name)
            if season is None or not self._season_in_range(season):
                continue
            for candidate in sorted(season_path.iterdir()):
                if candidate.is_dir():
                    week = _coerce_int(candidate.name)
                    path = candidate
                else:
                    if candidate.suffix.lower() not in {".csv", ".parquet"}:
                        continue
                    week = _coerce_int(candidate.stem)
                    path = candidate
                if week is None or not self._week_in_range(week):
                    continue
                key = (season, week)
                if key in self._processed:
                    continue
                yield BacktestTask(season=season, week=week, path=path)

    def _discover_seasonal(self) -> Iterable[BacktestTask]:
        config = self._config
        for candidate in sorted(config.snapshot_root.iterdir()):
            season = _coerce_int(candidate.name if candidate.is_dir() else candidate.stem)
            if season is None or not self._season_in_range(season):
                continue
            key = (season, None)
            if key in self._processed:
                continue
            yield BacktestTask(season=season, week=None, path=candidate)

    def _season_in_range(self, season: int) -> bool:
        config = self._config
        if config.start_season is not None and season < config.start_season:
            return False
        if config.end_season is not None and season > config.end_season:
            return False
        return True

    def _week_in_range(self, week: int) -> bool:
        config = self._config
        if config.start_week is not None and week < config.start_week:
            return False
        if config.end_week is not None and week > config.end_week:
            return False
        return True


def _coerce_int(value: str) -> int | None:
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:  # pragma: no cover - defensive
        return None


__all__ = [
    "Scheduler",
    "ScheduledJob",
    "BacktestScheduleConfig",
    "BacktestOrchestrator",
]

