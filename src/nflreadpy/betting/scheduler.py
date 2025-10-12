"""Asynchronous scheduler with jitter, retries, and graceful shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


AsyncCallable = Callable[[], Awaitable[Any]]


@dataclasses.dataclass(slots=True)
class ScheduledJob:
    """Representation of a coroutine executed at an interval."""

    name: str
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


__all__ = ["Scheduler", "ScheduledJob"]

