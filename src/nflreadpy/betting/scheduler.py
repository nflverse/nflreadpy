"""Asynchronous scheduler with jitter, retries, and graceful shutdown."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
from collections.abc import Awaitable, Callable
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
        attempt = 0
        while not stop_event.is_set():
            try:
                await self.action()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - logging path
                attempt += 1
                logger.exception("Scheduled job %s failed: %s", self.name, exc)
                if attempt > self.retries:
                    attempt = 0
                else:
                    backoff = self.retry_backoff * attempt
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                    except asyncio.TimeoutError:
                        continue
                    return
            delay = self.interval
            if self.jitter:
                delta = random.uniform(-self.jitter, self.jitter)
                delay = max(0.0, delay + delta)
            if delay == 0:
                await asyncio.sleep(0)
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
            return


class Scheduler:
    """Manage a collection of scheduled asynchronous jobs."""

    def __init__(self) -> None:
        self._jobs: list[ScheduledJob] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()

    def add_job(
        self,
        action: AsyncCallable,
        *,
        interval: float,
        jitter: float = 0.0,
        retries: int = 0,
        retry_backoff: float = 2.0,
        name: str | None = None,
    ) -> None:
        job_name = name or getattr(action, "__name__", "scheduled-job")
        self._jobs.append(
            ScheduledJob(
                name=job_name,
                action=action,
                interval=interval,
                jitter=jitter,
                retries=retries,
                retry_backoff=retry_backoff,
            )
        )

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
        try:
            await self._stop_event.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cancel running jobs and wait for them to exit."""

        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


__all__ = ["Scheduler", "ScheduledJob"]

