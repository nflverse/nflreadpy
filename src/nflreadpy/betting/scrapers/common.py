"""Shared async HTTP utilities for sportsbook scrapers."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from typing import Any, Mapping

from urllib import parse as urllib_parse
from urllib import request as urllib_request

try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    requests = None


class AsyncHTTPClient:
    """Very small async wrapper around :mod:`requests` for our scrapers."""

    def __init__(self, timeout: float | None = None) -> None:
        self._session = requests.Session() if requests else None
        self._timeout = timeout

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._request_json(url, params=params, headers=headers)
        )

    def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        if self._session:
            response = self._session.get(
                url, params=params, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
            return response.json()
        query_url = url
        if params:
            query = urllib_parse.urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            query_url = f"{url}{separator}{query}"
        req = urllib_request.Request(query_url, headers=dict(headers or {}))
        with urllib_request.urlopen(req, timeout=self._timeout) as response:
            body = response.read()
        return json.loads(body.decode("utf-8"))

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        if self._session:
            await loop.run_in_executor(None, self._session.close)

    def clone(self) -> "AsyncHTTPClient":
        """Return a copy sharing the same timeout configuration."""

        return AsyncHTTPClient(timeout=self._timeout)


class RateLimiter:
    """Simple asyncio-friendly rate limiter."""

    def __init__(self, requests_per_second: float | None) -> None:
        self._interval = 0.0
        if requests_per_second and requests_per_second > 0:
            self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait_time = self._interval - (now - self._last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.monotonic()


def deep_copy_payload(payload: Any) -> Any:
    """Utility to copy decoded JSON structures for safe reuse."""

    return copy.deepcopy(payload)
